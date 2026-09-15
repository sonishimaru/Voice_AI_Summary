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
    return merge_regions(regions, cfg.min_silence_ms, int(cfg.max_speech_s * 1000))


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
