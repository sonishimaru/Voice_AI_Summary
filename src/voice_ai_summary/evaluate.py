"""Measuring transcription accuracy: character error rate over a fixed set of clips.

An eval set is a JSON file of clips - a recording, a time range, and the correct text -
that stays put while ASR settings change around it. `vas eval build` seeds one from a
day's transcript (the text is the current output, so it must be corrected by hand before
the numbers mean anything); `vas eval run` re-transcribes those clips with the current
settings, or with an override, and reports CER against the reference.

Nothing here touches the network, and `run_eval` takes the backend so a caller can hold
two of them side by side and compare models.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .asr import ASRBackend
from .audio import SAMPLE_RATE, load_audio_16k
from .config import Config


@dataclass
class Clip:
    recording_id: int
    start_ms: int
    end_ms: int
    text: str

    @property
    def seconds(self) -> float:
        return (self.end_ms - self.start_ms) / 1000


@dataclass
class ClipResult:
    clip: Clip
    hypothesis: str
    edits: int

    @property
    def cer(self) -> float:
        """Character error rate. 0.0 is perfect; >1.0 means more edits than reference."""
        return self.edits / len(self.clip.text) if self.clip.text else float(bool(self.hypothesis))


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance. Character-level, which is the right unit for Japanese."""
    if a == b:
        return 0
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def normalize_for_cer(text: str) -> str:
    """Drop whitespace and the punctuation ASR sprinkles unpredictably."""
    return "".join(c for c in text if not c.isspace() and c not in "、。,.!?！？「」『』()（）")


def load_clips(path: Path) -> list[Clip]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Clip(**c) for c in data["clips"]]


def save_clips(path: Path, clips: list[Clip]) -> None:
    payload = {
        "note": "`text` starts as ASR output - correct it by hand before trusting the CER.",
        "clips": [vars(c) for c in clips],
    }
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_clips(conn: sqlite3.Connection, day_start_utc: str, day_end_utc: str) -> list[Clip]:
    """One clip per segment that produced text, with that text as the starting reference."""
    rows = conn.execute(
        """
        SELECT s.recording_id AS recording_id, s.start_ms AS start_ms, s.end_ms AS end_ms,
               group_concat(COALESCE(u.raw_text, u.text), '') AS text
        FROM segments s
        JOIN utterances u ON u.segment_id = s.id
        JOIN recordings r ON r.id = s.recording_id
        WHERE u.abs_start_utc >= ? AND u.abs_start_utc < ?
        GROUP BY s.id
        ORDER BY s.recording_id, s.start_ms
        """,
        (day_start_utc, day_end_utc),
    ).fetchall()
    return [Clip(r["recording_id"], r["start_ms"], r["end_ms"], r["text"] or "") for r in rows]


def run_eval(
    conn: sqlite3.Connection, cfg: Config, clips: list[Clip], backend: ASRBackend
) -> list[ClipResult]:
    """Re-transcribe each clip with `backend` and score it against the reference."""
    results: list[ClipResult] = []
    by_recording: dict[int, list[Clip]] = {}
    for clip in clips:
        by_recording.setdefault(clip.recording_id, []).append(clip)

    for recording_id, group in by_recording.items():
        row = conn.execute(
            "SELECT storage_path FROM recordings WHERE id = ?", (recording_id,)
        ).fetchone()
        if row is None:
            continue
        samples = load_audio_16k(cfg.paths.store / row["storage_path"])
        for clip in group:
            chunk = samples[
                int(clip.start_ms * SAMPLE_RATE / 1000) : int(clip.end_ms * SAMPLE_RATE / 1000)
            ]
            text = "".join(
                u.text for u in backend.transcribe(chunk, language=cfg.asr.language)
            ).strip()
            ref, hyp = normalize_for_cer(clip.text), normalize_for_cer(text)
            results.append(ClipResult(clip, text, edit_distance(ref, hyp)))
    return results


def aggregate(results: list[ClipResult]) -> dict:
    """Corpus CER (total edits over total reference characters) plus a few counts."""
    ref_chars = sum(len(normalize_for_cer(r.clip.text)) for r in results)
    edits = sum(r.edits for r in results)
    return {
        "clips": len(results),
        "seconds": sum(r.clip.seconds for r in results),
        "ref_chars": ref_chars,
        "edits": edits,
        "cer": edits / ref_chars if ref_chars else 0.0,
        "empty": sum(1 for r in results if not r.hypothesis.strip()),
    }
