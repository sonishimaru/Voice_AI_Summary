"""ASR backend option plumbing (no model download)."""

from __future__ import annotations

import numpy as np
import pytest

from voice_ai_summary.asr import (
    MAX_HOTWORD_CHARS,
    FasterWhisperBackend,
    Utterance,
    _merged_hotwords,
    drop_consecutive_repeats,
    get_backend,
    normalize_samples,
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


def test_normalize_samples_lifts_a_quiet_chunk_and_leaves_a_loud_one() -> None:
    quiet = np.full(1000, 0.1, dtype=np.float32)
    assert float(np.abs(normalize_samples(quiet, "peak")).max()) == pytest.approx(0.95, abs=0.01)

    faint = np.full(1000, 0.01, dtype=np.float32)
    assert float(np.sqrt((normalize_samples(faint, "rms") ** 2).mean())) == pytest.approx(
        0.05, abs=0.001
    )

    # Normalizing only ever adds gain - a chunk already at full scale is left alone.
    loud = np.full(1000, 1.0, dtype=np.float32)
    assert np.array_equal(normalize_samples(loud, "peak"), loud)
    assert np.array_equal(normalize_samples(loud, "rms"), loud)
    assert np.array_equal(
        normalize_samples(np.full(1000, 0.8, dtype=np.float32), "none"),
        np.full(1000, 0.8, dtype=np.float32),
    )


def test_normalize_samples_caps_the_gain_on_near_silence() -> None:
    """Amplifying room tone to speech level just gives the decoder noise to invent over,
    so the gain is capped instead of reaching the target."""
    room_tone = np.full(1000, 0.002, dtype=np.float32)
    assert float(np.abs(normalize_samples(room_tone, "peak")).max()) == pytest.approx(0.04)


def test_asr_normalize_can_be_overridden_per_source() -> None:
    """Same shape as `VadConfig.threshold_by_source`/`for_source`: one global `normalize`
    can't serve both the quiet mic track and the already-loud system track."""
    cfg = Config()
    cfg.asr.normalize = "none"
    cfg.asr.normalize_by_source = {"mac_mic": "rms"}

    assert cfg.asr.for_source("mac_mic").normalize == "rms"
    assert cfg.asr.for_source("mac_system").normalize == "none"
    assert cfg.asr.for_source("mac_system") is cfg.asr
    assert cfg.asr.for_source(None) is cfg.asr


def test_faster_whisper_transcribe_uses_the_per_source_normalize_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The override must actually reach the backend's normalization call, not just sit
    on the config object."""
    modes_seen: list[str] = []

    def fake_normalize(samples, mode):
        modes_seen.append(mode)
        return samples

    monkeypatch.setattr("voice_ai_summary.asr.normalize_samples", fake_normalize)

    cfg = Config()
    cfg.asr.normalize = "none"
    cfg.asr.normalize_by_source = {"mac_mic": "rms"}
    backend = FasterWhisperBackend(cfg)
    backend._model = _FakeModel()

    backend.transcribe(np.zeros(16000, dtype=np.float32), language="ja", source="mac_mic")
    backend.transcribe(np.zeros(16000, dtype=np.float32), language="ja", source="mac_system")
    backend.transcribe(np.zeros(16000, dtype=np.float32), language="ja")

    assert modes_seen == ["rms", "none", "none"]


def test_faster_whisper_passes_decoder_thresholds_at_library_defaults() -> None:
    """Defaults must equal faster-whisper's own, read from its `transcribe` signature, so
    leaving these unset changes nothing."""
    import inspect

    from faster_whisper import WhisperModel

    cfg = Config()
    backend = FasterWhisperBackend(cfg)
    model = _FakeModel()
    backend._model = model

    backend.transcribe(np.zeros(16000, dtype=np.float32), language="ja")

    sig = inspect.signature(WhisperModel.transcribe)
    assert model.kwargs["no_speech_threshold"] == sig.parameters["no_speech_threshold"].default
    assert model.kwargs["log_prob_threshold"] == sig.parameters["log_prob_threshold"].default
    assert (
        model.kwargs["compression_ratio_threshold"]
        == sig.parameters["compression_ratio_threshold"].default
    )
    assert cfg.asr.no_speech_threshold == sig.parameters["no_speech_threshold"].default
    assert cfg.asr.log_prob_threshold == sig.parameters["log_prob_threshold"].default
    assert (
        cfg.asr.compression_ratio_threshold == sig.parameters["compression_ratio_threshold"].default
    )


def test_faster_whisper_passes_configured_decoder_thresholds() -> None:
    cfg = Config()
    cfg.asr.no_speech_threshold = 0.3
    cfg.asr.log_prob_threshold = -0.5
    cfg.asr.compression_ratio_threshold = 2.0
    backend = FasterWhisperBackend(cfg)
    model = _FakeModel()
    backend._model = model

    backend.transcribe(np.zeros(16000, dtype=np.float32), language="ja")

    assert model.kwargs["no_speech_threshold"] == 0.3
    assert model.kwargs["log_prob_threshold"] == -0.5
    assert model.kwargs["compression_ratio_threshold"] == 2.0


class _RepeatingModel:
    """Simulates a repetition-loop decode: the same line comes back several times."""

    def __init__(self, texts: list[str]) -> None:
        self._texts = texts
        self.kwargs: dict = {}

    def transcribe(self, samples, **kwargs):
        self.kwargs = kwargs
        segs = [_Seg(float(i), float(i + 1), text) for i, text in enumerate(self._texts)]
        return iter(segs), None


def test_faster_whisper_drops_consecutive_repeats_by_default() -> None:
    cfg = Config()
    backend = FasterWhisperBackend(cfg)
    backend._model = _RepeatingModel(["同じ文", "同じ文", "同じ文", "別の文"])

    out = backend.transcribe(np.zeros(16000, dtype=np.float32), language="ja")

    assert [u.text for u in out] == ["同じ文", "別の文"]


def test_faster_whisper_repeat_filter_can_be_disabled() -> None:
    cfg = Config()
    cfg.asr.drop_repeated_utterances = False
    backend = FasterWhisperBackend(cfg)
    backend._model = _RepeatingModel(["同じ文", "同じ文"])

    out = backend.transcribe(np.zeros(16000, dtype=np.float32), language="ja")

    assert [u.text for u in out] == ["同じ文", "同じ文"]


def _utt(text: str) -> Utterance:
    return Utterance(t_start_ms=0, t_end_ms=1000, text=text, lang="ja", avg_logprob=None)


def test_drop_consecutive_repeats_collapses_only_adjacent_duplicates() -> None:
    utterances = [_utt("こんにちは"), _utt("こんにちは"), _utt("さようなら"), _utt("こんにちは")]

    out = drop_consecutive_repeats(utterances)

    # The final "こんにちは" is not adjacent to the first two, so it survives - a person
    # genuinely repeating themselves later must not be treated as a decoder loop.
    assert [u.text for u in out] == ["こんにちは", "さようなら", "こんにちは"]


def test_drop_consecutive_repeats_strips_before_comparing() -> None:
    out = drop_consecutive_repeats([_utt("こんにちは"), _utt(" こんにちは ")])
    assert len(out) == 1


def test_drop_consecutive_repeats_handles_empty_and_single_element_lists() -> None:
    assert drop_consecutive_repeats([]) == []
    one = [_utt("x")]
    assert drop_consecutive_repeats(one) == one
