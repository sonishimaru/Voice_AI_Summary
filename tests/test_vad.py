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
