"""ASR backends: faster-whisper (real transcription) and a deterministic fake for tests."""

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
        self.name = f"faster-whisper:{cfg.asr.model}"

    def _get_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self._cfg.asr.model,
                device=self._cfg.asr.device,
                compute_type=self._cfg.asr.compute_type,
            )
        return self._model

    def transcribe(self, samples16k: np.ndarray, *, language: str | None) -> list[Utterance]:
        model = self._get_model()
        segments, _info = model.transcribe(
            samples16k,
            language=language,
            beam_size=self._cfg.asr.beam_size,
            vad_filter=False,
            condition_on_previous_text=False,
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
    if os.environ.get("VAS_ASR_BACKEND") == "fake":
        return FakeBackend()
    return FasterWhisperBackend(cfg)
