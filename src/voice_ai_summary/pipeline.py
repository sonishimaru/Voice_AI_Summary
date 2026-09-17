"""Recording → VAD → ASR → SQLite."""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from . import vad as vad_module
from .asr import ASRBackend
from .audio import SAMPLE_RATE, duration_ms, load_audio_16k
from .config import Config
from .db import average_processed_duration_ms, recent_realtime_factor, transaction, utcnow_iso
from .ingest import parse_inbox_name
from .timeutil import fmt_hm, to_local

logger = logging.getLogger(__name__)

_SOURCE_SPEAKERS = {"mac_mic": "me", "mac_system": "other"}

# A claim older than this is treated as abandoned and can be picked up by another
# caller. Must comfortably exceed the slowest realistic single-recording decode: the
# user measures ~12 minutes to transcribe a 15-minute recording on CPU, and a
# `launchctl kickstart -k` restart (routine here) can kill the worker mid-decode and
# leave `claimed_at` set with nothing left to clear it. 45 minutes gives generous
# headroom over that 12-minute measurement - for a slower machine or a longer
# recording - while still being far short of "forever", so a genuinely abandoned claim
# doesn't strand the row for days.
CLAIM_TIMEOUT_S = 45 * 60


def speaker_for_source(source: str) -> str:
    return _SOURCE_SPEAKERS.get(source, "unknown")


def _parse_iso(iso_utc: str) -> datetime:
    return datetime.strptime(iso_utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


def _fmt_iso(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _abs_start_utc(started_at_utc: str, t_start_ms: int) -> str:
    started = datetime.strptime(started_at_utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    return (started + timedelta(milliseconds=t_start_ms)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _clear_transcript(conn: sqlite3.Connection, recording_ids: list[int]) -> None:
    """Delete `segments`/`utterances` rows for the given recordings.

    The `utterances_ad` trigger in `schema.sql` mirrors each deleted row into
    `utterances_fts` (an external-content FTS5 table), so this is the one place that
    removes utterances - anything that drops or replaces a recording's transcript must
    go through here rather than issuing its own `DELETE FROM utterances`, or the FTS
    index goes stale. Caller is expected to be inside a `transaction(conn)` block.
    """
    if not recording_ids:
        return
    placeholders = ",".join("?" for _ in recording_ids)
    conn.execute(f"DELETE FROM utterances WHERE recording_id IN ({placeholders})", recording_ids)
    conn.execute(f"DELETE FROM segments WHERE recording_id IN ({placeholders})", recording_ids)


def process_recording(
    conn: sqlite3.Connection, cfg: Config, recording_id: int, backend: ASRBackend
) -> int:
    """Run VAD + ASR for one recording, insert segments/utterances. Returns utterance count.

    Idempotent: any segments/utterances already stored for this recording (from an
    earlier run) are deleted, in the same transaction, before the new ones are
    inserted - so re-processing the same recording replaces its transcript instead of
    appending a second copy of it.
    """
    row = conn.execute("SELECT * FROM recordings WHERE id = ?", (recording_id,)).fetchone()
    if row is None:
        raise ValueError(f"no such recording: {recording_id}")

    if row["audio_deleted_at"] is not None:
        # Terminal state: retention already removed this recording's audio file, so
        # there is nothing left to decode. Recorded the same way an ordinary decode
        # failure is (error set, claim cleared) so callers see one consistent shape.
        message = (
            f"audio was deleted by retention on {row['audio_deleted_at']}; cannot re-transcribe"
        )
        with transaction(conn):
            conn.execute(
                "UPDATE recordings SET error = ?, claimed_at = NULL WHERE id = ?",
                (message, recording_id),
            )
        raise RuntimeError(message)

    try:
        audio_path = cfg.paths.store / row["storage_path"]
        samples = load_audio_16k(audio_path)
        length_ms = duration_ms(samples)
        speaker = speaker_for_source(row["source"])

        # Measured with time.monotonic() (immune to wall-clock adjustments) around VAD +
        # the ASR decode loop only - this is the work whose cost actually depends on the
        # backend (faster-whisper on CPU vs mlx on the Apple GPU) and is what
        # `throughput`'s realtime factor reports on. Deliberately NOT included: decoding
        # the source audio file just above (container/codec I/O, not ASR) and the DB
        # write transaction just below (brief, and the same cost regardless of backend).
        decode_start = time.monotonic()
        regions = vad_module.detect_speech(samples, cfg.vad.for_source(row["source"]))

        # Transcribe first, write second: a 15-minute recording takes minutes to decode,
        # and holding the write lock for that long locks out the worker (or `vas
        # reprocess`) running alongside - it fails with "database is locked".
        decoded = []
        for region in regions:
            start_sample = int(region.start_ms * SAMPLE_RATE / 1000)
            end_sample = int(region.end_ms * SAMPLE_RATE / 1000)
            chunk = samples[start_sample:end_sample]
            decoded.append(
                (
                    region,
                    backend.transcribe(chunk, language=cfg.asr.language, source=row["source"]),
                )
            )
        processing_ms = int(round((time.monotonic() - decode_start) * 1000))

        utterance_count = 0
        with transaction(conn):
            _clear_transcript(conn, [recording_id])
            conn.execute(
                "UPDATE recordings SET duration_ms = ? WHERE id = ?", (length_ms, recording_id)
            )
            for region, utterances in decoded:
                cur = conn.execute(
                    "INSERT INTO segments (recording_id, start_ms, end_ms, speech_prob) "
                    "VALUES (?, ?, ?, ?)",
                    (recording_id, region.start_ms, region.end_ms, region.prob),
                )
                segment_id = cur.lastrowid

                for utt in utterances:
                    text = utt.text.strip()
                    if not text:
                        continue
                    t_start_ms = region.start_ms + utt.t_start_ms
                    t_end_ms = region.start_ms + utt.t_end_ms
                    abs_start_utc = _abs_start_utc(row["started_at_utc"], t_start_ms)
                    conn.execute(
                        """
                        INSERT INTO utterances
                            (recording_id, segment_id, t_start_ms, t_end_ms, abs_start_utc,
                             text, lang, asr_model, avg_logprob, speaker)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            recording_id,
                            segment_id,
                            t_start_ms,
                            t_end_ms,
                            abs_start_utc,
                            text,
                            utt.lang,
                            backend.name,
                            utt.avg_logprob,
                            speaker,
                        ),
                    )
                    utterance_count += 1
            conn.execute(
                # Clear the claim alongside processed_at: this is the terminal state, so
                # nothing needs to keep owning the row. processing_ms rides along too -
                # it's the same "this recording is done" write.
                "UPDATE recordings SET processed_at = ?, processing_ms = ?, claimed_at = NULL"
                " WHERE id = ?",
                (utcnow_iso(), processing_ms, recording_id),
            )
        return utterance_count
    except Exception as exc:
        with transaction(conn):
            conn.execute(
                # Same here: an errored recording is terminal until `retry_failed` clears
                # the error, so the claim must not outlive the failure.
                "UPDATE recordings SET error = ?, claimed_at = NULL WHERE id = ?",
                (str(exc)[:500], recording_id),
            )
        raise
    except BaseException:
        # Not an ordinary decode failure but a shutdown (`worker._Stop`, deliberately a
        # `BaseException` so it lands here instead of the `except Exception` above -
        # see its docstring) or another non-Exception interrupt. This recording was not
        # given a fair chance to transcribe, so it must not be recorded as failed and
        # must stay eligible for the next attempt: clear only the claim, leaving `error`
        # and `processed_at` exactly as they were found (both still NULL - this row was
        # pending). Otherwise a `launchctl kickstart -k` restart mid-decode - a routine,
        # one-click path - would strand the row under its live claim until
        # `CLAIM_TIMEOUT_S` (45 minutes) expires before anyone picks it up again.
        with transaction(conn):
            conn.execute("UPDATE recordings SET claimed_at = NULL WHERE id = ?", (recording_id,))
        raise


def _claim_recording(conn: sqlite3.Connection, recording_id: int) -> bool:
    """Atomically take ownership of one pending recording before decoding it.

    The worker and the `process_pending` MCP tool poll the same database, so between
    the `SELECT` that finds a candidate and the (multi-minute) decode there is a window
    for a second caller to pick up the same id and transcribe it a second time. This is
    the single conditional `UPDATE` that closes that window: the WHERE clause re-checks
    the very preconditions the caller's `SELECT` used, plus a claim check, so only one
    of two racing callers can flip the row. `rowcount == 0` means another live claim (or
    a state change) beat this one to it - the caller must skip the row, not decode it.
    A claim older than `CLAIM_TIMEOUT_S` counts as abandoned and is reclaimable.
    """
    now_dt = datetime.now(UTC)
    now = now_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    stale_before = (now_dt - timedelta(seconds=CLAIM_TIMEOUT_S)).strftime("%Y-%m-%dT%H:%M:%SZ")
    cur = conn.execute(
        "UPDATE recordings SET claimed_at = ? WHERE id = ? AND processed_at IS NULL "
        "AND error IS NULL AND (claimed_at IS NULL OR claimed_at < ?)",
        (now, recording_id, stale_before),
    )
    conn.commit()
    return cur.rowcount > 0


def process_pending(
    conn: sqlite3.Connection,
    cfg: Config,
    backend: ASRBackend,
    limit: int | None = None,
    *,
    within: tuple[str, str] | None = None,
) -> int:
    """Process unprocessed, error-free recordings, oldest first. Returns count actually
    processed by this call - a recording already claimed by another caller is skipped,
    not counted as a failure.

    `within` is a `(start_utc, end_utc)` half-open bound on `started_at_utc`, for callers
    that must not spend their budget elsewhere: the nightly digest catches up the day it
    is about to summarize, and draining days-old audio instead would leave that day
    untranscribed however long it ran.
    """
    sql = "SELECT id FROM recordings WHERE processed_at IS NULL AND error IS NULL"
    params: tuple[str, ...] = ()
    if within is not None:
        sql += " AND started_at_utc >= ? AND started_at_utc < ?"
        params = within
    sql += " ORDER BY started_at_utc ASC"
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    ids = [r["id"] for r in conn.execute(sql, params).fetchall()]

    processed = 0
    for rec_id in ids:
        # Claim one recording right before processing it, not the whole batch upfront -
        # a crash between claims must not strand ids this caller never got to.
        if not _claim_recording(conn, rec_id):
            continue
        try:
            process_recording(conn, cfg, rec_id, backend)
            processed += 1
        except Exception:
            logger.exception("failed to process recording %s", rec_id)
    return processed


def backlog_eta(conn: sqlite3.Connection) -> str:
    """One-line, human-readable estimate of how long the pending backlog will take to clear.

    Pending recordings don't have a `duration_ms` of their own yet (it's only set once
    `process_recording` runs), so the pending audio total is projected from the average
    length of already-processed recordings, then divided by `recent_realtime_factor` -
    this session's actual recent decode speed - to get a time estimate. Says plainly
    that there's no timing data yet, instead of inventing a number, when either figure
    is unavailable (e.g. nothing has ever been processed).
    """
    pending = conn.execute(
        "SELECT COUNT(*) AS n FROM recordings WHERE processed_at IS NULL AND error IS NULL"
    ).fetchone()["n"]
    if pending == 0:
        return "backlog: none pending."

    avg_ms = average_processed_duration_ms(conn)
    factor = recent_realtime_factor(conn)
    if avg_ms is None or factor is None or factor <= 0:
        return (
            f"backlog: {pending} recording(s) pending; no timing data yet to estimate how "
            "long clearing it will take (process at least one recording, e.g. via "
            "process_pending, to get a reading)."
        )

    est_audio_min = pending * avg_ms / 60_000
    est_processing_min = est_audio_min / factor
    return (
        f"backlog: {pending} recording(s) pending, an estimated {est_audio_min:.1f} min of "
        "audio (projected from the average recent recording length); at the recent "
        f"{factor:.2f}x realtime factor, clearing it should take about "
        f"{est_processing_min:.1f} min."
    )


def retry_failed(conn: sqlite3.Connection, cfg: Config) -> list[int]:
    """Clear the error flag on failed recordings so `process_pending` picks them up again.

    A recording ingested while the recorder still had it open lands in the store under its
    `.part` name; once the recorder has closed it the file is valid, so it is renamed here.

    Skips (and leaves errored) any recording whose audio retention already deleted -
    `process_recording` would just set the same "audio was deleted" error right back,
    so clearing it here would only make the row flicker pending before failing again.
    """
    rows = conn.execute(
        "SELECT id, storage_path, audio_deleted_at FROM recordings WHERE error IS NOT NULL"
    ).fetchall()
    ids: list[int] = []
    for row in rows:
        if row["audio_deleted_at"] is not None:
            continue
        storage_path = row["storage_path"]
        if storage_path.endswith(".part"):
            src = cfg.paths.store / storage_path
            dst = src.with_name(src.name[: -len(".part")])
            if src.exists():
                src.rename(dst)
            storage_path = storage_path[: -len(".part")]
        conn.execute(
            # Clear claimed_at too: without this a retried recording stays invisible to
            # `process_pending` until the stale-claim timeout expires, even though it is
            # pending again right now.
            "UPDATE recordings SET error = NULL, processed_at = NULL, claimed_at = NULL,"
            " storage_path = ? WHERE id = ?",
            (storage_path, row["id"]),
        )
        ids.append(row["id"])
    conn.commit()
    return ids


def reset_recordings(conn: sqlite3.Connection, recording_ids: list[int]) -> int:
    """Drop transcripts for the given recordings and mark them pending again.

    Used to re-transcribe after changing the ASR model, vocabulary or VAD chunking.
    Summaries are keyed by transcript content, so they recompute on the next digest.

    Skips any recording whose audio retention already deleted - there is nothing left
    to re-transcribe it from - and logs which ids were skipped. Returns the count
    actually reset, which can be smaller than `len(recording_ids)`.
    """
    if not recording_ids:
        return 0
    placeholders = ",".join("?" for _ in recording_ids)
    rows = conn.execute(
        f"SELECT id, audio_deleted_at FROM recordings WHERE id IN ({placeholders})",
        recording_ids,
    ).fetchall()
    resettable_ids = [r["id"] for r in rows if r["audio_deleted_at"] is None]
    for r in rows:
        if r["audio_deleted_at"] is not None:
            logger.warning(
                "skipping reset of recording %s: audio was deleted by retention on %s",
                r["id"],
                r["audio_deleted_at"],
            )
    if not resettable_ids:
        return 0
    reset_placeholders = ",".join("?" for _ in resettable_ids)
    with transaction(conn):
        _clear_transcript(conn, resettable_ids)
        conn.execute(
            # claimed_at = NULL for the same reason as in retry_failed: a re-queued
            # recording must be visible to process_pending immediately, not after the
            # stale-claim timeout.
            f"UPDATE recordings SET processed_at = NULL, error = NULL, claimed_at = NULL"
            f" WHERE id IN ({reset_placeholders})",
            resettable_ids,
        )
    return len(resettable_ids)


def delete_recording(
    conn: sqlite3.Connection,
    recording_id: int,
    *,
    cfg: Config | None = None,
    delete_audio: bool = False,
) -> bool:
    """Permanently delete one recording's row along with its segments/utterances.

    By default does NOT touch the audio file on disk - callers that want that removed
    too must either pass `cfg` and `delete_audio=True`, or do it themselves. When both
    are given, the audio file is unlinked (a missing file is fine) before the row is
    deleted, so a crash between the two leaves an orphaned file rather than a row
    pointing at nothing.

    Returns False (no-op) if the recording does not exist. Uses `_clear_transcript` so
    `utterances_fts` stays consistent, same as `reset_recordings` and `process_recording`.
    """
    row = conn.execute(
        "SELECT storage_path FROM recordings WHERE id = ?", (recording_id,)
    ).fetchone()
    if row is None:
        return False

    if delete_audio and cfg is not None and row["storage_path"]:
        try:
            (cfg.paths.store / row["storage_path"]).unlink()
        except FileNotFoundError:
            pass

    with transaction(conn):
        _clear_transcript(conn, [recording_id])
        conn.execute("DELETE FROM recordings WHERE id = ?", (recording_id,))
    return True


def recordings_overlapping(
    conn: sqlite3.Connection,
    start_utc: str,
    end_utc: str,
    *,
    default_duration_ms: int,
) -> list[sqlite3.Row]:
    """Recordings whose `[started_at_utc, started_at_utc + duration)` window overlaps
    the half-open `[start_utc, end_utc)` range - a recording ending exactly at
    `start_utc` does NOT overlap.

    A recording with no `duration_ms` yet (still pending, or errored before it could be
    measured) is assumed to last `default_duration_ms` - typically the recorder's
    rotation interval (`cfg.recorder.rotation_minutes`), the same assumption
    `delete_range` uses for the recordings it finds this way.

    The SQL below is a wide net over candidates (no recording is expected to run
    anywhere near a full day), not the exact overlap test - that needs each row's actual
    duration, and is applied in Python over the candidate set.
    """
    window_start = _fmt_iso(_parse_iso(start_utc) - timedelta(hours=24))
    candidates = conn.execute(
        "SELECT * FROM recordings WHERE started_at_utc < ? AND started_at_utc >= ?"
        " ORDER BY started_at_utc",
        (end_utc, window_start),
    ).fetchall()
    out = []
    for row in candidates:
        dur = row["duration_ms"] if row["duration_ms"] is not None else default_duration_ms
        row_end = _fmt_iso(_parse_iso(row["started_at_utc"]) + timedelta(milliseconds=dur))
        if row["started_at_utc"] < end_utc and row_end > start_utc:
            out.append(row)
    return out


@dataclass
class DeleteRangeReport:
    recordings: list[dict] = field(default_factory=list)
    inbox_files: list[str] = field(default_factory=list)
    days: list[str] = field(default_factory=list)
    episode_summaries_removed: int = 0
    day_digests_removed: list[str] = field(default_factory=list)
    dry_run: bool = True

    def summary(self, tz: str) -> str:
        lines = []
        if self.dry_run:
            lines.append(
                "DRY RUN - nothing was deleted. Call again with dry_run=False to actually delete."
            )
        lines.append(
            f"deletion is per RECORDING: the {len(self.recordings)} whole recording(s) "
            f"covering the requested range go, not just the requested minutes within them "
            f"(local time, {tz}):"
        )
        for r in self.recordings:
            audio_state = "audio present" if r["audio_present"] else "audio already gone"
            lines.append(
                f"  recording {r['id']} ({r['source']}) local start {r['local_start']}"
                f" duration~{r['duration_min']:.1f}min utterances={r['utterances']}"
                f" ({audio_state})"
            )
        if self.inbox_files:
            lines.append(f"{len(self.inbox_files)} not-yet-ingested inbox file(s) also covered:")
            for name in self.inbox_files:
                lines.append(f"  {name}")
        lines.append(f"affected local day(s): {', '.join(self.days) if self.days else '(none)'}")
        removed_digests = (
            ", ".join(self.day_digests_removed) if self.day_digests_removed else "(none)"
        )
        lines.append(
            f"episode summaries removed: {self.episode_summaries_removed}; "
            f"day digest(s) removed: {removed_digests} "
            "- these day(s) need `rebuild_day` (or `vas digest`) run again to get a digest back."
        )
        return "\n".join(lines)


def delete_range(
    conn: sqlite3.Connection,
    cfg: Config,
    start_utc: str,
    end_utc: str,
    *,
    dry_run: bool = True,
) -> DeleteRangeReport:
    """Permanently delete every recording overlapping `[start_utc, end_utc)`, along with
    their audio, the not-yet-ingested inbox files covering the same window, the episode/
    day summaries that depended on them, and the affected days' digest files - then
    rebuild those days' episodes so nothing points at deleted utterances.

    Deletion is per-recording: a recording is atomic here (it came from one continuous
    audio file), so asking to delete a five-minute slice removes the *whole* recording(s)
    that slice falls in, per `recordings_overlapping`'s overlap rule.

    `dry_run=True` (the default) only reads - the returned report is an accurate preview
    and nothing on disk or in the database changes. `dry_run=False` performs the
    deletion: DB writes (summaries, then recordings - audio + row, via `delete_recording`,
    so `_clear_transcript` keeps `utterances_fts` consistent) happen first; only after
    that do the pure filesystem steps happen (inbox files/sidecars, rebuilding episodes,
    the digest files). That order means a worker mid-decode of a recording just deleted
    fails on file-not-found (its audio, or its row) and its final `UPDATE ... WHERE id=?`
    then simply matches no row - not a crash, just a no-op. `.part` files (the recorder's
    still-open marker) are never touched.
    """
    tz = cfg.summarize.timezone
    default_duration_ms = cfg.recorder.rotation_minutes * 60_000

    overlapping = recordings_overlapping(
        conn, start_utc, end_utc, default_duration_ms=default_duration_ms
    )

    rec_infos: list[dict] = []
    days_set: set[str] = set()
    for row in overlapping:
        dur = row["duration_ms"] if row["duration_ms"] is not None else default_duration_ms
        utt_count = conn.execute(
            "SELECT COUNT(*) AS n FROM utterances WHERE recording_id = ?", (row["id"],)
        ).fetchone()["n"]
        rec_infos.append(
            {
                "id": row["id"],
                "source": row["source"],
                "local_start": fmt_hm(row["started_at_utc"], tz),
                "duration_min": dur / 60_000,
                "utterances": utt_count,
                "audio_present": row["audio_deleted_at"] is None,
            }
        )
        end_dt = _parse_iso(row["started_at_utc"]) + timedelta(milliseconds=dur)
        days_set.add(to_local(row["started_at_utc"], tz).strftime("%Y-%m-%d"))
        days_set.add(to_local(_fmt_iso(end_dt), tz).strftime("%Y-%m-%d"))
    days = sorted(days_set)

    inbox_files: list[str] = []
    if cfg.paths.inbox.is_dir():
        for path in sorted(cfg.paths.inbox.iterdir()):
            if not path.is_file() or path.suffix in (".json", ".part") or path.name.startswith("."):
                continue
            parsed = parse_inbox_name(path)
            if parsed is None:
                continue
            row_start = parsed["started_at_utc"]
            row_end = _fmt_iso(_parse_iso(row_start) + timedelta(milliseconds=default_duration_ms))
            if row_start < end_utc and row_end > start_utc:
                inbox_files.append(path.name)

    # What this would drop from `summaries` - computed whether or not this is a dry run
    # (read-only), so the preview matches what a real run would remove.
    from .episodes import build_episodes, episode_transcript
    from .summarize import _content_key

    episode_ids = [
        r["id"]
        for r in conn.execute(
            "SELECT id FROM episodes WHERE started_at_utc < ? AND ended_at_utc > ?",
            (end_utc, start_utc),
        ).fetchall()
    ]
    episode_keys = sorted({_content_key(episode_transcript(conn, eid, tz)) for eid in episode_ids})

    episode_summaries_removed = 0
    if episode_keys:
        placeholders = ",".join("?" for _ in episode_keys)
        episode_summaries_removed = conn.execute(
            f"SELECT COUNT(*) AS n FROM summaries"
            f" WHERE scope='episode' AND scope_key IN ({placeholders})",
            episode_keys,
        ).fetchone()["n"]

    day_digests_removed: list[str] = []
    if days:
        placeholders = ",".join("?" for _ in days)
        day_digests_removed = sorted(
            r["scope_key"]
            for r in conn.execute(
                f"SELECT scope_key FROM summaries"
                f" WHERE scope='day' AND scope_key IN ({placeholders})",
                days,
            ).fetchall()
        )

    report = DeleteRangeReport(
        recordings=rec_infos,
        inbox_files=inbox_files,
        days=days,
        episode_summaries_removed=episode_summaries_removed,
        day_digests_removed=day_digests_removed,
        dry_run=dry_run,
    )

    if dry_run:
        return report

    with transaction(conn):
        if episode_keys:
            placeholders = ",".join("?" for _ in episode_keys)
            conn.execute(
                f"DELETE FROM summaries WHERE scope='episode' AND scope_key IN ({placeholders})",
                episode_keys,
            )
        if days:
            placeholders = ",".join("?" for _ in days)
            conn.execute(
                f"DELETE FROM summaries WHERE scope='day' AND scope_key IN ({placeholders})",
                days,
            )
        for info in rec_infos:
            delete_recording(conn, info["id"], cfg=cfg, delete_audio=True)
        if episode_ids:
            # `delete_recording` above (via `_clear_transcript`) already removed every
            # utterance these episodes had - clean up the now-empty episode rows too, or
            # they'd linger forever with nothing pointing at them. Safe even for an
            # episode that also had utterances outside the deleted recordings: those
            # utterances' `episode_id` just goes to NULL (`ON DELETE SET NULL`), and
            # `build_episodes` below regenerates the day's episodes from scratch anyway.
            placeholders = ",".join("?" for _ in episode_ids)
            conn.execute(f"DELETE FROM episodes WHERE id IN ({placeholders})", episode_ids)

    # Filesystem cleanup, deliberately after the DB transaction above has committed.
    for name in inbox_files:
        path = cfg.paths.inbox / name
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        sidecar = path.with_suffix(".json")
        try:
            sidecar.unlink()
        except FileNotFoundError:
            pass

    for day in days:
        build_episodes(conn, cfg, day)
        try:
            (cfg.paths.digests / f"{day}.md").unlink()
        except FileNotFoundError:
            pass
        mirror = cfg.paths.digest_mirror
        if mirror is not None:
            try:
                (mirror / f"{day}.md").unlink()
            except FileNotFoundError:
                pass

    return report
