"""Tests for the CER harness. `run_eval` uses the fake backend, so no model is loaded."""

from __future__ import annotations

import numpy as np

from voice_ai_summary.asr import FakeBackend
from voice_ai_summary.evaluate import (
    Clip,
    aggregate,
    edit_distance,
    load_clips,
    normalize_for_cer,
    run_eval,
    save_clips,
)


def test_edit_distance() -> None:
    assert edit_distance("", "") == 0
    assert edit_distance("行太郎", "行太郎") == 0
    assert edit_distance("ギョウ太郎", "行太郎") == 3
    assert edit_distance("あいう", "") == 3


def test_normalize_for_cer_drops_punctuation_and_space() -> None:
    assert normalize_for_cer("こんにちは、元気ですか？") == "こんにちは元気ですか"
    assert normalize_for_cer(" 西丸 です 。") == "西丸です"


def test_clips_roundtrip(tmp_path) -> None:
    clips = [Clip(recording_id=1, start_ms=0, end_ms=2000, text="こんにちは")]
    path = tmp_path / "eval.json"
    save_clips(path, clips)
    assert load_clips(path) == clips


def test_run_eval_scores_against_the_reference(vas, monkeypatch) -> None:
    cfg, conn = vas
    conn.execute(
        """
        INSERT INTO recordings(id, source, device_id, started_at_utc, tz_offset, duration_ms,
                               sha256, storage_path, original_name, ingested_at)
        VALUES (1, 'mac_mic', 'dev', '2026-09-15T00:00:00Z', '+00:00', 2000,
                'a', 'x.wav', 'x.wav', '2026-09-15T00:00:00Z')
        """
    )
    conn.commit()
    monkeypatch.setattr(
        "voice_ai_summary.evaluate.load_audio_16k", lambda path: np.zeros(32000, dtype=np.float32)
    )

    clips = [
        Clip(recording_id=1, start_ms=0, end_ms=1000, text="こんにちは"),
        Clip(recording_id=1, start_ms=1000, end_ms=2000, text="さようなら"),
    ]
    results = run_eval(conn, cfg, clips, FakeBackend(["こんにちは", "さような"]))

    assert [r.edits for r in results] == [0, 1]
    stats = aggregate(results)
    assert stats["clips"] == 2
    assert stats["ref_chars"] == 10
    assert stats["cer"] == 0.1
    assert stats["empty"] == 0
