"""Recording → VAD → ASR → SQLite."""

from __future__ import annotations

import logging
import sqlite3
from datetime import UTC, datetime, timedelta

from . import vad as vad_module
from .asr import ASRBackend
from .audio import SAMPLE_RATE, duration_ms, load_audio_16k
from .config import Config
from .db import transaction, utcnow_iso

logger = logging.getLogger(__name__)

_SOURCE_SPEAKERS = {"mac_mic": "me", "mac_system": "other"}


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

        regions = vad_module.detect_speech(samples, cfg.vad.for_source(row["source"]))

        # Transcribe first, write second: a 15-minute recording takes minutes to decode,
        # and holding the write lock for that long locks out the worker (or `vas
        # reprocess`) running alongside - it fails with "database is locked".
        decoded = []
        for region in regions:
            start_sample = int(region.start_ms * SAMPLE_RATE / 1000)
            end_sample = int(region.end_ms * SAMPLE_RATE / 1000)
            chunk = samples[start_sample:end_sample]
            decoded.append((region, backend.transcribe(chunk, language=cfg.asr.language)))

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
                "UPDATE recordings SET processed_at = ? WHERE id = ?",
                (utcnow_iso(), recording_id),
            )
        return utterance_count
    except Exception as exc:
        with transaction(conn):
            conn.execute(
                "UPDATE recordings SET error = ? WHERE id = ?",
                (str(exc)[:500], recording_id),
            )
        raise


def process_pending(
    conn: sqlite3.Connection, cfg: Config, backend: ASRBackend, limit: int | None = None
) -> int:
    """Process all unprocessed, error-free recordings, oldest first. Returns count processed."""
    sql = (
        "SELECT id FROM recordings WHERE processed_at IS NULL AND error IS NULL "
        "ORDER BY started_at_utc ASC"
    )
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    ids = [r["id"] for r in conn.execute(sql).fetchall()]

    processed = 0
    for rec_id in ids:
        try:
            process_recording(conn, cfg, rec_id, backend)
            processed += 1
        except Exception:
            logger.exception("failed to process recording %s", rec_id)
    return processed


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
            "UPDATE recordings SET error = NULL, processed_at = NULL, storage_path = ?"
            " WHERE id = ?",
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
            f"UPDATE recordings SET processed_at = NULL, error = NULL WHERE id IN ({placeholders})",
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
