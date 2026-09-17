"""ASR backends: faster-whisper (CPU), mlx-whisper (Apple GPU) and a deterministic fake."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from .config import Config


@dataclass
class Utterance:
    t_start_ms: int
    t_end_ms: int
    text: str
    lang: str | None
    avg_logprob: float | None


class ASRBackend(Protocol):
    name: str

    def transcribe(
        self, samples16k: np.ndarray, *, language: str | None, source: str | None = None
    ) -> list[Utterance]: ...


# A vocabulary prompt costs transcription on kotoba-whisper-v2.0. Measured on three
# 30-second chunks of real audio, utterances returned per chunk against prompt length:
#
#   0 terms (0 chars): 10 / 4 / 8      12 terms (60 chars): 8 / 1 / 1
#   8 terms (35 chars): 7 / 2 / 8      16 terms (79 chars): 0 / 0 / 0
#
# It degrades from the first term and collapses to silence well before Whisper's own
# 223-token prompt ceiling, through `hotwords` and `initial_prompt` alike. Cap hard at a
# length the measurement showed to be survivable; `use_glossary_hotwords` stays off.
MAX_HOTWORD_CHARS = 50


def _merged_hotwords(cfg: Config, extra_hotwords: list[str]) -> list[str]:
    """`cfg.asr.hotwords` extended with the user's glossary hotwords, deduplicated.

    Truncated to `MAX_HOTWORD_CHARS` worth of terms: config hotwords come first, so
    curating `[asr] hotwords` is how to pick what survives a large glossary.
    """
    seen: set[str] = set()
    merged: list[str] = []
    size = 0
    for word in (*cfg.asr.hotwords, *extra_hotwords):
        if not word or word in seen:
            continue
        if size + len(word) + 1 > MAX_HOTWORD_CHARS:
            continue  # skip this one; a single long term must not drop the whole glossary
        size += len(word) + 1
        seen.add(word)
        merged.append(word)
    return merged


# Speech sits around this RMS once levelled; the gain cap keeps a silent chunk from being
# amplified into noise the decoder then hallucinates over.
_TARGET_RMS = 0.05
_TARGET_PEAK = 0.95
_MAX_GAIN = 20.0


def normalize_samples(samples: np.ndarray, mode: str) -> np.ndarray:
    """Scale a chunk to a workable level. `mode` is "none", "peak" or "rms"."""
    if mode == "none" or len(samples) == 0:
        return samples
    if mode == "peak":
        current = float(np.abs(samples).max())
        target = _TARGET_PEAK
    elif mode == "rms":
        current = float(np.sqrt((samples.astype("float64") ** 2).mean()))
        target = _TARGET_RMS
    else:
        raise ValueError(f"unknown [asr] normalize mode: {mode}")
    if current <= 0:
        return samples
    gain = min(target / current, _MAX_GAIN)
    if gain <= 1.0:
        return samples
    return np.clip(samples * gain, -1.0, 1.0).astype("float32")


def drop_consecutive_repeats(
    utterances: list[Utterance], *, max_gap_ms: int = 1000
) -> list[Utterance]:
    """Drop an utterance whose stripped text equals the one immediately before it, but
    only when it follows that predecessor within `max_gap_ms` (see
    `AsrConfig.repeat_gap_max_ms`).

    Whisper's classic failure mode on a decode call is a repetition loop: the same short
    line emitted several times in a row, back-to-back, inside one window. This filter
    only ever compares an utterance to its immediate predecessor *within the list from a
    single `transcribe()` call* - it must never be applied across separate calls, because
    a person genuinely repeating themselves minutes apart is real speech, not a decoder
    artifact, and two recordings of the same audio producing identical text is a
    different bug, already handled elsewhere (idempotent reprocessing).

    The timestamp bound is required *now that `vad.pack_regions` exists*: packing joins
    several originally-separate VAD speech regions, absorbing the real silence between
    them, into one decode call - so two genuinely separate utterances (e.g. two distinct
    "はい" several seconds apart) can now arrive adjacent in the same call's output. Only
    a decoder repetition loop looks like back-to-back output with (near) zero elapsed
    time between repeats; a real pause between separately-detected utterances is bounded
    below by `vad.merge_regions`' `merge_gap_ms`, since two regions are only left distinct
    (rather than merged into one) when they are at least that far apart. Comparing only
    adjacent entries, never all-pairs, and bounding by real elapsed time, not just position
    in the list, is what keeps a legitimate "yes, yes, okay" - whether back-to-back or
    minutes apart - from ever being collapsed further than the loop itself, or dropped as
    if it were one.
    """
    out: list[Utterance] = []
    prev_text: str | None = None
    prev_end_ms: int | None = None
    for utt in utterances:
        text = utt.text.strip()
        gap_ms = None if prev_end_ms is None else utt.t_start_ms - prev_end_ms
        if out and text == prev_text and gap_ms is not None and gap_ms <= max_gap_ms:
            continue
        out.append(utt)
        prev_text = text
        prev_end_ms = utt.t_end_ms
    return out


def _prompt_for(cfg: Config, hotwords: list[str]) -> str | None:
    """Same formula as `AsrConfig.prompt`, but over a (possibly glossary-extended) list."""
    parts = [cfg.asr.initial_prompt.strip()] if cfg.asr.initial_prompt.strip() else []
    if hotwords:
        parts.append("、".join(hotwords) + "。")
    return " ".join(parts) or None


class FasterWhisperBackend:
    """Lazily loaded faster-whisper model."""

    def __init__(self, cfg: Config, *, extra_hotwords: list[str] | None = None) -> None:
        self._cfg = cfg
        self._extra_hotwords = extra_hotwords or []
        self._model = None
        self.name = f"faster-whisper:{cfg.asr.resolved_model}"

    def _get_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self._cfg.asr.resolved_model,
                device=self._cfg.asr.device,
                compute_type=self._cfg.asr.compute_type,
            )
        return self._model

    def transcribe(
        self, samples16k: np.ndarray, *, language: str | None, source: str | None = None
    ) -> list[Utterance]:
        model = self._get_model()
        asr = self._cfg.asr.for_source(source)
        hotwords = _merged_hotwords(self._cfg, self._extra_hotwords)
        segments, _info = model.transcribe(
            normalize_samples(samples16k, asr.normalize),
            language=language,
            beam_size=asr.beam_size,
            vad_filter=False,
            condition_on_previous_text=False,
            initial_prompt=asr.initial_prompt.strip() or None,
            hotwords=" ".join(hotwords) or None,
            no_speech_threshold=asr.no_speech_threshold,
            log_prob_threshold=asr.log_prob_threshold,
            compression_ratio_threshold=asr.compression_ratio_threshold,
        )
        utterances = [
            Utterance(
                t_start_ms=int(round(seg.start * 1000)),
                t_end_ms=int(round(seg.end * 1000)),
                text=seg.text.strip(),
                lang=language,
                avg_logprob=seg.avg_logprob,
            )
            for seg in segments
        ]
        if asr.drop_repeated_utterances:
            utterances = drop_consecutive_repeats(utterances, max_gap_ms=asr.repeat_gap_max_ms)
        return utterances


class MlxWhisperBackend:
    """mlx-whisper: runs Whisper on the Apple Silicon GPU (macOS only)."""

    def __init__(self, cfg: Config, *, extra_hotwords: list[str] | None = None) -> None:
        self._cfg = cfg
        self._extra_hotwords = extra_hotwords or []
        self.name = f"mlx-whisper:{cfg.asr.resolved_model}"

    def transcribe(
        self, samples16k: np.ndarray, *, language: str | None, source: str | None = None
    ) -> list[Utterance]:
        import mlx_whisper

        asr = self._cfg.asr.for_source(source)
        hotwords = _merged_hotwords(self._cfg, self._extra_hotwords)
        result = mlx_whisper.transcribe(
            normalize_samples(samples16k, asr.normalize),
            path_or_hf_repo=asr.resolved_model,
            language=language,
            initial_prompt=_prompt_for(self._cfg, hotwords),
            condition_on_previous_text=False,
            verbose=None,
            # mlx-whisper names this `logprob_threshold` (no underscore before "prob"),
            # unlike faster-whisper's `log_prob_threshold` - confirmed by reading
            # mlx_whisper.transcribe's own signature, same defaults as faster-whisper.
            no_speech_threshold=asr.no_speech_threshold,
            logprob_threshold=asr.log_prob_threshold,
            compression_ratio_threshold=asr.compression_ratio_threshold,
        )
        utterances = [
            Utterance(
                t_start_ms=int(round(seg["start"] * 1000)),
                t_end_ms=int(round(seg["end"] * 1000)),
                text=seg["text"].strip(),
                lang=language,
                avg_logprob=seg.get("avg_logprob"),
            )
            for seg in result["segments"]
        ]
        if asr.drop_repeated_utterances:
            utterances = drop_consecutive_repeats(utterances, max_gap_ms=asr.repeat_gap_max_ms)
        return utterances


class FakeBackend:
    """Deterministic backend for tests: one utterance per call, covering the whole input."""

    name = "fake"

    def __init__(self, texts: list[str] | None = None) -> None:
        self._texts = list(texts or ["テスト"])
        self._i = 0

    def transcribe(
        self, samples16k: np.ndarray, *, language: str | None, source: str | None = None
    ) -> list[Utterance]:
        from .audio import duration_ms

        text = self._texts[self._i % len(self._texts)]
        self._i += 1
        return [
            Utterance(
                t_start_ms=0,
                t_end_ms=duration_ms(samples16k),
                text=text,
                lang=language,
                avg_logprob=None,
            )
        ]


def get_backend(cfg: Config) -> ASRBackend:
    backend = os.environ.get("VAS_ASR_BACKEND") or cfg.asr.backend
    if backend == "fake":
        return FakeBackend()

    from .glossary import load_glossary

    extra_hotwords = load_glossary(cfg).hotwords() if cfg.asr.use_glossary_hotwords else []
    if backend == "mlx":
        return MlxWhisperBackend(cfg, extra_hotwords=extra_hotwords)
    if backend == "faster-whisper":
        return FasterWhisperBackend(cfg, extra_hotwords=extra_hotwords)
    raise ValueError(f"unknown ASR backend: {backend!r} (faster-whisper | mlx)")
