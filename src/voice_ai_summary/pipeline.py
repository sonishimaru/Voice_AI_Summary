"""Recording → VAD → ASR → SQLite."""

from __future__ import annotations

import logging
import sqlite3
import time
from datetime import UTC, datetime, timedelta

from . import vad as vad_module
from .asr import ASRBackend
from .audio import SAMPLE_RATE, duration_ms, load_audio_16k
from .config import Config
from .db import average_processed_duration_ms, recent_realtime_factor, transaction, utcnow_iso

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
    """
    rows = conn.execute(
        "SELECT id, storage_path FROM recordings WHERE error IS NOT NULL"
    ).fetchall()
    ids: list[int] = []
    for row in rows:
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
    """
    if not recording_ids:
        return 0
    placeholders = ",".join("?" for _ in recording_ids)
    with transaction(conn):
        _clear_transcript(conn, recording_ids)
        conn.execute(
            # claimed_at = NULL for the same reason as in retry_failed: a re-queued
            # recording must be visible to process_pending immediately, not after the
            # stale-claim timeout.
            f"UPDATE recordings SET processed_at = NULL, error = NULL, claimed_at = NULL"
            f" WHERE id IN ({placeholders})",
            recording_ids,
        )
    return len(recording_ids)


def delete_recording(conn: sqlite3.Connection, recording_id: int) -> bool:
    """Permanently delete one recording's row along with its segments/utterances.

    Does NOT touch the audio file on disk - callers that want that removed too must do
    it themselves. Returns False (no-op) if the recording does not exist. Uses
    `_clear_transcript` so `utterances_fts` stays consistent, same as `reset_recordings`
    and `process_recording`.
    """
    with transaction(conn):
        row = conn.execute("SELECT id FROM recordings WHERE id = ?", (recording_id,)).fetchone()
        if row is None:
            return False
        _clear_transcript(conn, [recording_id])
        conn.execute("DELETE FROM recordings WHERE id = ?", (recording_id,))
    return True
