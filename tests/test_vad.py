"""Tests for VAD region merging and (lightly) the real Silero model."""

from __future__ import annotations

import numpy as np

from voice_ai_summary.config import VadConfig
from voice_ai_summary.vad import SpeechRegion, detect_speech, merge_regions


def test_merge_regions_bridges_small_gaps() -> None:
    regions = [
        SpeechRegion(0, 1000, 0.9),
        SpeechRegion(1100, 2000, 0.8),  # 100ms gap, below min_gap_ms
    ]
    merged = merge_regions(regions, min_gap_ms=500, max_len_ms=100_000)
    assert merged == [SpeechRegion(0, 2000, 0.9)]


def test_merge_regions_keeps_large_gaps_separate() -> None:
    regions = [
        SpeechRegion(0, 1000, 0.9),
        SpeechRegion(2000, 3000, 0.8),  # 1000ms gap, above min_gap_ms
    ]
    merged = merge_regions(regions, min_gap_ms=500, max_len_ms=100_000)
    assert merged == regions


def test_merge_regions_splits_long_spans() -> None:
    regions = [SpeechRegion(0, 2500, 0.5)]
    merged = merge_regions(regions, min_gap_ms=500, max_len_ms=1000)
    assert [(r.start_ms, r.end_ms) for r in merged] == [(0, 1000), (1000, 2000), (2000, 2500)]


def test_merge_regions_empty() -> None:
    assert merge_regions([], min_gap_ms=500, max_len_ms=1000) == []


def test_merge_regions_sorts_out_of_order_input() -> None:
    regions = [SpeechRegion(2000, 3000, None), SpeechRegion(0, 1000, None)]
    merged = merge_regions(regions, min_gap_ms=100, max_len_ms=100_000)
    assert [(r.start_ms, r.end_ms) for r in merged] == [(0, 1000), (2000, 3000)]


def test_detect_speech_runs_real_silero_model() -> None:
    """Silero may or may not fire on a pure tone; just verify it runs offline and returns a list."""
    sr = 16000
    rng = np.random.default_rng(0)
    silence = np.zeros(2 * sr, dtype=np.float32)
    t = np.linspace(0, 1, sr, endpoint=False)
    tone = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    noise = (0.05 * rng.standard_normal(sr)).astype(np.float32)
    signal = np.concatenate([silence, tone + noise, silence])

    result = detect_speech(signal, VadConfig())
    assert isinstance(result, list)
    for region in result:
        assert isinstance(region, SpeechRegion)
        assert region.start_ms < region.end_ms


def test_detect_speech_merges_across_short_pauses(monkeypatch) -> None:
    """Sub-2 s pauses between phrases must not split a sentence into fragments."""
    from voice_ai_summary import vad as vad_module
    from voice_ai_summary.config import VadConfig

    fake_regions = [
        {"start": 0, "end": 16000},  # 0-1 s
        {"start": 24000, "end": 40000},  # 1.5-2.5 s (0.5 s pause)
        {"start": 56000, "end": 72000},  # 3.5-4.5 s (1.0 s pause)
        {"start": 160000, "end": 176000},  # 10-11 s (5.5 s pause → new chunk)
    ]
    monkeypatch.setattr(vad_module, "_run_silero", lambda samples, cfg: fake_regions)
    regions = vad_module.detect_speech(np.zeros(16000 * 12, dtype=np.float32), VadConfig())
    assert [(r.start_ms, r.end_ms) for r in regions] == [(0, 4500), (10000, 11000)]
