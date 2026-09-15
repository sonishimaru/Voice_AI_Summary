"""ASR backend option plumbing (no model download)."""

from __future__ import annotations

import numpy as np

from voice_ai_summary.asr import FasterWhisperBackend, get_backend
from voice_ai_summary.config import DEFAULT_MLX_MODEL, Config


class _Seg:
    def __init__(self, start: float, end: float, text: str) -> None:
        self.start, self.end, self.text, self.avg_logprob = start, end, text, -0.2


class _FakeModel:
    def __init__(self) -> None:
        self.kwargs: dict = {}

    def transcribe(self, samples, **kwargs):
        self.kwargs = kwargs
        return iter([_Seg(0.0, 1.2, " 安田さんに送ります ")]), None


def test_faster_whisper_passes_vocabulary_and_prompt() -> None:
    cfg = Config()
    cfg.asr.hotwords = ["安田さん", "西丸"]
    cfg.asr.initial_prompt = "社内会議の録音です。"
    backend = FasterWhisperBackend(cfg)
    model = _FakeModel()
    backend._model = model

    out = backend.transcribe(np.zeros(16000, dtype=np.float32), language="ja")

    assert model.kwargs["hotwords"] == "安田さん 西丸"
    assert model.kwargs["initial_prompt"] == "社内会議の録音です。 安田さん、西丸。"
    assert model.kwargs["condition_on_previous_text"] is False
    assert out[0].text == "安田さんに送ります"
    assert (out[0].t_start_ms, out[0].t_end_ms) == (0, 1200)


def test_prompt_is_none_without_vocabulary() -> None:
    assert Config().asr.prompt is None


def test_mlx_backend_swaps_default_model(monkeypatch) -> None:
    monkeypatch.delenv("VAS_ASR_BACKEND", raising=False)
    cfg = Config()
    cfg.asr.backend = "mlx"
    backend = get_backend(cfg)
    assert backend.name == f"mlx-whisper:{DEFAULT_MLX_MODEL}"
    cfg.asr.model = "mlx-community/whisper-large-v3"
    assert get_backend(cfg).name == "mlx-whisper:mlx-community/whisper-large-v3"
