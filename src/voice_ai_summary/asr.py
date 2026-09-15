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


class FasterWhisperBackend:
    """Lazily loaded faster-whisper model."""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
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
        segments, _info = model.transcribe(
            samples16k,
            language=language,
            beam_size=asr.beam_size,
            vad_filter=False,
            condition_on_previous_text=False,
            initial_prompt=asr.prompt,
            hotwords=" ".join(asr.hotwords) or None,
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

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self.name = f"mlx-whisper:{cfg.asr.resolved_model}"

    def transcribe(self, samples16k: np.ndarray, *, language: str | None) -> list[Utterance]:
        import mlx_whisper

        asr = self._cfg.asr
        result = mlx_whisper.transcribe(
            samples16k,
            path_or_hf_repo=asr.resolved_model,
            language=language,
            initial_prompt=asr.prompt,
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
    if backend == "mlx":
        return MlxWhisperBackend(cfg)
    if backend == "faster-whisper":
        return FasterWhisperBackend(cfg)
    raise ValueError(f"unknown ASR backend: {backend!r} (faster-whisper | mlx)")
