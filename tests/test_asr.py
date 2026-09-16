"""ASR backend option plumbing (no model download)."""

from __future__ import annotations

import numpy as np

from voice_ai_summary.asr import (
    MAX_HOTWORD_CHARS,
    FasterWhisperBackend,
    _merged_hotwords,
    get_backend,
)
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
    # Vocabulary goes through the hotwords slot only: faster-whisper applies it to every
    # window, and repeating it in initial_prompt would eat the decoder's context.
    assert model.kwargs["initial_prompt"] == "社内会議の録音です。"
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


def test_get_backend_leaves_the_glossary_out_of_the_decoder(monkeypatch, tmp_path) -> None:
    """A vocabulary prompt silences kotoba-whisper (see `MAX_HOTWORD_CHARS`), so the
    glossary reaches the correction pass but not the decoder unless asked for."""
    monkeypatch.setenv("VAS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("VAS_ASR_BACKEND", "faster-whisper")

    from voice_ai_summary.config import load_config
    from voice_ai_summary.glossary import Glossary, Term, save_glossary

    cfg = load_config()
    cfg.asr.hotwords = ["安田さん"]
    save_glossary(cfg, Glossary(terms=[Term(term="西丸", aliases=["にしまる"])]))

    backend = get_backend(cfg)
    model = _FakeModel()
    backend._model = model
    backend.transcribe(np.zeros(16000, dtype=np.float32), language="ja")

    assert model.kwargs["hotwords"] == "安田さん"
    assert model.kwargs["initial_prompt"] is None


def test_get_backend_can_opt_into_glossary_hotwords(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("VAS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("VAS_ASR_BACKEND", "faster-whisper")

    from voice_ai_summary.config import load_config
    from voice_ai_summary.glossary import Glossary, Term, save_glossary

    cfg = load_config()
    cfg.asr.hotwords = ["安田さん"]
    cfg.asr.use_glossary_hotwords = True
    save_glossary(cfg, Glossary(terms=[Term(term="西丸", aliases=["にしまる"])]))

    backend = get_backend(cfg)
    backend._model = model = _FakeModel()
    backend.transcribe(np.zeros(16000, dtype=np.float32), language="ja")

    assert model.kwargs["hotwords"] == "安田さん 西丸 にしまる"


def test_hotwords_are_capped_to_what_the_decoder_will_read() -> None:
    """Whisper only conditions on ~223 tokens of prompt; a 300-term glossary would be
    silently cut, so cap it here and keep the configured terms at the front."""
    cfg = Config()
    cfg.asr.hotwords = ["安田さん"]
    glossary_terms = [f"用語{i:03d}" for i in range(300)]

    merged = _merged_hotwords(cfg, glossary_terms)

    assert merged[0] == "安田さん"
    assert len(" ".join(merged)) <= MAX_HOTWORD_CHARS
    assert len(merged) < len(glossary_terms)
