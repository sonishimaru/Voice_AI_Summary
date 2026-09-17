"""Tests for VAD region merging and (lightly) the real Silero model."""

from __future__ import annotations

import numpy as np

from voice_ai_summary.config import VadConfig
from voice_ai_summary.vad import SpeechRegion, detect_speech, merge_regions, pack_regions


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
    """Sub-2 s pauses between phrases must not split a sentence into fragments.

    The 5.5 s pause before the last chunk also exceeds the default `pack_max_gap_ms`
    (5000 ms), so the packing pass (run after merging) leaves these two chunks separate
    too - this test's expected output is unchanged by packing.
    """
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


def test_detect_speech_packs_regions_within_gap_and_cap(monkeypatch) -> None:
    """A pause just inside `pack_max_gap_ms` gets absorbed into one packed region."""
    from voice_ai_summary import vad as vad_module
    from voice_ai_summary.config import VadConfig

    fake_regions = [
        {"start": 0, "end": 16000},  # 0-1 s
        {"start": 24000, "end": 40000},  # 1.5-2.5 s (0.5 s pause, merged)
        # 4.5 s pause after the merged (0, 2500) region: below pack_max_gap_ms (5s)
        {"start": 112000, "end": 128000},  # 7-8 s
    ]
    monkeypatch.setattr(vad_module, "_run_silero", lambda samples, cfg: fake_regions)
    cfg = VadConfig(pack_max_gap_ms=5000)
    regions = vad_module.detect_speech(np.zeros(16000 * 10, dtype=np.float32), cfg)
    assert [(r.start_ms, r.end_ms) for r in regions] == [(0, 8000)]


def test_pack_regions_joins_small_gap() -> None:
    """A gap smaller than `max_gap_ms` that still fits the cap is absorbed into one span."""
    regions = [
        SpeechRegion(0, 1000, 0.9),
        SpeechRegion(3000, 4000, 0.7),  # 2 s gap, below the 5 s cap used here
    ]
    packed = pack_regions(regions, max_span_ms=30_000, max_gap_ms=5000)
    assert len(packed) == 1
    assert (packed[0].start_ms, packed[0].end_ms) == (0, 4000)
    # duration-weighted mean of the two 1000 ms regions' probabilities
    assert packed[0].prob == 0.8


def test_pack_regions_keeps_large_gap_separate() -> None:
    """A gap larger than `max_gap_ms` is not absorbed, even though it would fit the cap."""
    regions = [
        SpeechRegion(0, 1000, 0.9),
        SpeechRegion(7000, 8000, 0.7),  # 6 s gap, above the 5 s cap
    ]
    packed = pack_regions(regions, max_span_ms=30_000, max_gap_ms=5000)
    assert [(r.start_ms, r.end_ms) for r in packed] == [(0, 1000), (7000, 8000)]


def test_pack_regions_stops_at_max_span() -> None:
    """Three regions that would together exceed the cap pack into two, not one."""
    regions = [
        SpeechRegion(0, 400, 1.0),
        SpeechRegion(450, 700, 1.0),  # 50 ms gap; combined span (0, 700) fits the 700 ms cap
        SpeechRegion(750, 1000, 1.0),  # 50 ms gap, but (0, 1000) would exceed the cap
    ]
    packed = pack_regions(regions, max_span_ms=700, max_gap_ms=100)
    assert [(r.start_ms, r.end_ms) for r in packed] == [(0, 700), (750, 1000)]


def test_pack_regions_leaves_oversized_region_untouched() -> None:
    """A single region already at or over `max_span_ms` passes through exactly as it is."""
    region = SpeechRegion(0, 5000, 0.42)
    packed = pack_regions([region], max_span_ms=1000, max_gap_ms=5000)
    assert packed == [region]


def test_pack_regions_disabled_when_max_gap_zero() -> None:
    """`max_gap_ms = 0` disables packing: the input comes back unchanged."""
    regions = [SpeechRegion(0, 1000, 0.9), SpeechRegion(1200, 2000, 0.8)]
    packed = pack_regions(regions, max_span_ms=30_000, max_gap_ms=0)
    assert packed == regions


def test_pack_regions_empty() -> None:
    assert pack_regions([], max_span_ms=30_000, max_gap_ms=5000) == []


def test_pack_regions_single_region_passthrough() -> None:
    region = SpeechRegion(100, 200, 0.5)
    assert pack_regions([region], max_span_ms=30_000, max_gap_ms=5000) == [region]


def test_pack_regions_cuts_region_count_by_large_factor() -> None:
    """Packing should sharply cut the number of ASR calls for a realistic conversation.

    20 speech regions of 2 s each, spaced 5 s apart (a 3 s pause between phrases - typical
    output from `merge_regions` with the default `merge_gap_ms`), span 97 s in total.
    Every 3 s gap is well under the default `pack_max_gap_ms` (5 s), so consecutive
    regions keep getting absorbed until the packed span would exceed `max_speech_s`
    (30 s here), at which point a new packed region starts. That works out to 6 original
    regions per packed region (their span is 27 s, and a 7th would push it to 32 s), so
    20 regions pack into 4 - a 5x reduction in the number of (fixed-cost, 30 s-window)
    ASR calls this recording needs.
    """
    regions = [SpeechRegion(i * 5000, i * 5000 + 2000, 0.8) for i in range(20)]
    packed = pack_regions(regions, max_span_ms=30_000, max_gap_ms=5000)
    assert len(packed) == 4
    assert len(regions) / len(packed) == 5.0
    for region in packed:
        assert region.end_ms - region.start_ms <= 30_000


def test_pack_regions_invariant_contains_members_and_stays_sorted_nonoverlapping() -> None:
    """Every packed span contains the original regions it was built from and the packed
    list itself stays sorted and non-overlapping, same as its input.
    """
    regions = [SpeechRegion(i * 5000, i * 5000 + 2000, 0.8) for i in range(20)]
    packed = pack_regions(regions, max_span_ms=30_000, max_gap_ms=5000)

    # sorted, non-overlapping
    for a, b in zip(packed, packed[1:], strict=False):
        assert a.end_ms <= b.start_ms

    # every original region is contained within some packed span
    for region in regions:
        assert any(p.start_ms <= region.start_ms and region.end_ms <= p.end_ms for p in packed)

    # every packed span's own bounds come from the original regions it absorbed
    assert packed[0].start_ms == regions[0].start_ms
    assert packed[-1].end_ms == regions[-1].end_ms
