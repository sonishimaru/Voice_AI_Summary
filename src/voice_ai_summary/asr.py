"""ASR backends: faster-whisper (CPU), mlx-whisper (Apple GPU) and a deterministic fake."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import numpy as np

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

    def transcribe(self, samples16k: np.ndarray, *, language: str | None) -> list[Utterance]: ...


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
        size += len(word) + 1
        if size > MAX_HOTWORD_CHARS:
            break
        seen.add(word)
        merged.append(word)
    return merged


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

    def transcribe(self, samples16k: np.ndarray, *, language: str | None) -> list[Utterance]:
        model = self._get_model()
        asr = self._cfg.asr
        hotwords = _merged_hotwords(self._cfg, self._extra_hotwords)
        segments, _info = model.transcribe(
            samples16k,
            language=language,
            beam_size=asr.beam_size,
            vad_filter=False,
            condition_on_previous_text=False,
            initial_prompt=asr.initial_prompt.strip() or None,
            hotwords=" ".join(hotwords) or None,
        )
        return [
            Utterance(
                t_start_ms=int(round(seg.start * 1000)),
                t_end_ms=int(round(seg.end * 1000)),
                text=seg.text.strip(),
                lang=language,
                avg_logprob=seg.avg_logprob,
            )
            for seg in segments
        ]


class MlxWhisperBackend:
    """mlx-whisper: runs Whisper on the Apple Silicon GPU (macOS only)."""

    def __init__(self, cfg: Config, *, extra_hotwords: list[str] | None = None) -> None:
        self._cfg = cfg
        self._extra_hotwords = extra_hotwords or []
        self.name = f"mlx-whisper:{cfg.asr.resolved_model}"

    def transcribe(self, samples16k: np.ndarray, *, language: str | None) -> list[Utterance]:
        import mlx_whisper

        asr = self._cfg.asr
        hotwords = _merged_hotwords(self._cfg, self._extra_hotwords)
        result = mlx_whisper.transcribe(
            samples16k,
            path_or_hf_repo=asr.resolved_model,
            language=language,
            initial_prompt=_prompt_for(self._cfg, hotwords),
            condition_on_previous_text=False,
            verbose=None,
        )
        return [
            Utterance(
                t_start_ms=int(round(seg["start"] * 1000)),
                t_end_ms=int(round(seg["end"] * 1000)),
                text=seg["text"].strip(),
                lang=language,
                avg_logprob=seg.get("avg_logprob"),
            )
            for seg in result["segments"]
        ]


class FakeBackend:
    """Deterministic backend for tests: one utterance per call, covering the whole input."""

    name = "fake"

    def __init__(self, texts: list[str] | None = None) -> None:
        self._texts = list(texts or ["テスト"])
        self._i = 0

    def transcribe(self, samples16k: np.ndarray, *, language: str | None) -> list[Utterance]:
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
