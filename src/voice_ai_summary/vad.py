"""Voice activity detection using faster-whisper's bundled Silero VAD (no torch needed)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .audio import SAMPLE_RATE
from .config import VadConfig


@dataclass
class SpeechRegion:
    start_ms: int
    end_ms: int
    prob: float | None


def _run_silero(samples16k: np.ndarray, cfg: VadConfig) -> list[dict]:
    """Isolated so tests can monkeypatch the model call."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    options = VadOptions(
        threshold=cfg.threshold,
        min_speech_duration_ms=cfg.min_speech_ms,
        min_silence_duration_ms=cfg.min_silence_ms,
        max_speech_duration_s=cfg.max_speech_s,
        speech_pad_ms=cfg.pad_ms,
    )
    return get_speech_timestamps(samples16k, vad_options=options, sampling_rate=SAMPLE_RATE)


def detect_speech(samples16k: np.ndarray, cfg: VadConfig) -> list[SpeechRegion]:
    """Detect speech regions in a mono 16 kHz float array."""
    timestamps = _run_silero(samples16k, cfg)
    regions = [
        SpeechRegion(
            start_ms=int(ts["start"] * 1000 / SAMPLE_RATE),
            end_ms=int(ts["end"] * 1000 / SAMPLE_RATE),
            prob=None,
        )
        for ts in timestamps
    ]
    max_len_ms = int(cfg.max_speech_s * 1000)
    merged = merge_regions(regions, cfg.merge_gap_ms, max_len_ms)
    return pack_regions(merged, max_span_ms=max_len_ms, max_gap_ms=cfg.pack_max_gap_ms)


def merge_regions(
    regions: list[SpeechRegion], min_gap_ms: int, max_len_ms: int
) -> list[SpeechRegion]:
    """Merge regions separated by < `min_gap_ms`; split anything longer than `max_len_ms`."""
    if not regions:
        return []
    ordered = sorted(regions, key=lambda r: r.start_ms)
    merged: list[SpeechRegion] = [ordered[0]]
    for region in ordered[1:]:
        last = merged[-1]
        if region.start_ms - last.end_ms < min_gap_ms:
            new_end = max(last.end_ms, region.end_ms)
            probs = [p for p in (last.prob, region.prob) if p is not None]
            prob = max(probs) if probs else None
            merged[-1] = SpeechRegion(start_ms=last.start_ms, end_ms=new_end, prob=prob)
        else:
            merged.append(region)

    split: list[SpeechRegion] = []
    for region in merged:
        span = region.end_ms - region.start_ms
        if span <= max_len_ms or max_len_ms <= 0:
            split.append(region)
            continue
        start = region.start_ms
        while start < region.end_ms:
            end = min(start + max_len_ms, region.end_ms)
            split.append(SpeechRegion(start_ms=start, end_ms=end, prob=region.prob))
            start = end
    return split


def pack_regions(
    regions: list[SpeechRegion], *, max_span_ms: int, max_gap_ms: int
) -> list[SpeechRegion]:
    """Greedily pack consecutive regions into larger contiguous spans of the original audio.

    Whisper pads every input up to its fixed (30 s) decoding window regardless of how much
    of it is actual speech, so transcribing a 3 s region costs the same compute as a 30 s
    one - most of the padding is wasted on silence. This packs several short, nearby
    regions from `merge_regions` into one contiguous `[start_ms, end_ms)` span (never a
    concatenation of disjoint pieces, so downstream timestamp arithmetic stays linear in
    the original timeline) up to `max_span_ms`, so one ASR call amortizes the window
    padding across what used to be several calls.

    Joining absorbs the silence between the packed regions - that's the point, since the
    window gets padded with silence either way - but only up to `max_gap_ms`: unbounded
    silence in the decoder's input invites Whisper repetition/hallucination loops. A
    region already at or over `max_span_ms` is left exactly as it is. `max_gap_ms <= 0`
    disables packing entirely.

    A packed region's `prob` is the duration-weighted mean of the probabilities of the
    regions it absorbed (probabilities of `None` are excluded from both the sum and the
    weight, so a run of unscored regions doesn't drag a known score toward 0); if none of
    the absorbed regions carry a probability, the result is `None`.
    """
    if not regions or max_gap_ms <= 0:
        return list(regions)

    ordered = sorted(regions, key=lambda r: r.start_ms)
    packed: list[SpeechRegion] = []
    i = 0
    n = len(ordered)
    while i < n:
        members = [ordered[i]]
        start = ordered[i].start_ms
        end = ordered[i].end_ms
        j = i + 1
        while j < n:
            candidate = ordered[j]
            gap = candidate.start_ms - end
            span = candidate.end_ms - start
            if gap > max_gap_ms or span > max_span_ms:
                break
            end = candidate.end_ms
            members.append(candidate)
            j += 1

        weighted = [(m.prob, m.end_ms - m.start_ms) for m in members if m.prob is not None]
        total_weight = sum(dur for _, dur in weighted)
        prob = sum(p * dur for p, dur in weighted) / total_weight if total_weight > 0 else None
        packed.append(SpeechRegion(start_ms=start, end_ms=end, prob=prob))
        i = j

    return packed
