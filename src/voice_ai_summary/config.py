"""Configuration: TOML file + environment variables (secrets are env-only)."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

from pydantic import BaseModel, Field

DEFAULT_CONFIG_PATH = Path("~/.config/voice-ai-summary/config.toml")


class PathsConfig(BaseModel):
    data_dir: Path = Path("~/Library/Application Support/VoiceAISummary")
    # Extra directory each daily digest is copied into. `data_dir` sits under
    # ~/Library/Application Support, which sandboxed readers (Cowork's device tools and
    # other agents) are refused access to; mirroring just the digests into an ordinary
    # folder gives them the day's summary without moving the recordings and the database
    # out of Application Support. Unset means no mirror.
    digest_mirror_dir: Path | None = None

    @property
    def root(self) -> Path:
        return self.data_dir.expanduser()

    @property
    def inbox(self) -> Path:
        return self.root / "inbox"

    @property
    def store(self) -> Path:
        return self.root / "store"

    @property
    def db_path(self) -> Path:
        return self.root / "vas.sqlite3"

    @property
    def digests(self) -> Path:
        return self.root / "digests"

    @property
    def digest_mirror(self) -> Path | None:
        return self.digest_mirror_dir.expanduser() if self.digest_mirror_dir else None


DEFAULT_FASTER_WHISPER_MODEL = "kotoba-tech/kotoba-whisper-v2.0-faster"
DEFAULT_MLX_MODEL = "mlx-community/whisper-large-v3-turbo"


class AsrConfig(BaseModel):
    backend: str = "faster-whisper"  # faster-whisper | mlx
    model: str = DEFAULT_FASTER_WHISPER_MODEL
    device: str = "auto"
    # "auto" lets CTranslate2 pick the fastest precision the device supports (int8 on
    # Apple Silicon CPU); "default" keeps the model's stored float32 and is several times
    # slower. Ignored by the mlx backend, which has its own precision.
    compute_type: str = "auto"
    language: str = "ja"
    beam_size: int = 5
    # Domain vocabulary (names, products, jargon) biases decoding toward these spellings.
    # Measured on kotoba-whisper-v2.0: every term added suppresses output, and past ~15
    # terms the model returns nothing at all. Keep this list very short, and see
    # `use_glossary_hotwords` before feeding it the whole glossary.
    hotwords: list[str] = Field(default_factory=list)
    # Feed the personal glossary to the decoder as hotwords. Off because it silences this
    # model; proper nouns are fixed by the Claude correction pass instead, which reads the
    # whole glossary with no length limit.
    use_glossary_hotwords: bool = False
    initial_prompt: str = ""
    # Level the chunk before decoding: "none" | "peak" | "rms". The mic track records far
    # quieter than the system track, and Whisper transcribes quiet speech worse.
    normalize: str = "none"
    # Per-source overrides, e.g. {"mac_mic": "rms"}: one global `normalize` can't serve
    # both tracks when only one of them is quiet. Same shape as
    # `VadConfig.threshold_by_source` - see `for_source` below.
    normalize_by_source: dict[str, str] = Field(default_factory=dict)
    # Suppress Whisper's classic repetition-loop failure: drop an utterance whose text is
    # identical to the one immediately before it, but only within a single decode call
    # (see `asr.drop_consecutive_repeats`). On by default since it targets a decoder bug,
    # not real speech; the switch exists in case it ever eats a genuine short repeated
    # phrase.
    drop_repeated_utterances: bool = True
    # How close (ms) a repeat's timestamp must follow its predecessor's for
    # `asr.drop_consecutive_repeats` to treat it as that decoder loop rather than real
    # speech. `vad.pack_regions` now joins separate VAD speech regions into one decode
    # call, absorbing up to `vad.pack_max_gap_ms` of real silence between them - so two
    # genuinely separate utterances can land adjacent in the same call. `vad.merge_regions`
    # only ever leaves two regions distinct (unmerged) when they are >= `vad.merge_gap_ms`
    # (2000ms default) apart, so any two utterances from genuinely separate regions are
    # guaranteed at least that much real silence between them once packed. A repetition
    # loop, by contrast, re-emits the same line inside a single decode window with (near)
    # zero elapsed time between repeats. Half of the default `merge_gap_ms` sits
    # comfortably below what a real pause looks like and comfortably above what a
    # same-window decoder repeat looks like.
    repeat_gap_max_ms: int = 1000
    # Decoder failure thresholds, passed straight through to faster-whisper/mlx-whisper.
    # Defaulted to each library's own default (read from `WhisperModel.transcribe` /
    # `mlx_whisper.transcribe`'s signatures, not measured) so leaving these unset changes
    # nothing. See `config.example.toml` for which direction suppresses hallucination.
    no_speech_threshold: float = 0.6
    log_prob_threshold: float = -1.0
    compression_ratio_threshold: float = 2.4

    def for_source(self, source: str | None) -> AsrConfig:
        """This config with `normalize` replaced by the override for `source`, if any."""
        override = self.normalize_by_source.get(source)
        return self if override is None else self.model_copy(update={"normalize": override})

    @property
    def resolved_model(self) -> str:
        """The MLX backend needs an MLX-converted model; swap the default when unset."""
        if self.backend == "mlx" and self.model == DEFAULT_FASTER_WHISPER_MODEL:
            return DEFAULT_MLX_MODEL
        return self.model

    @property
    def prompt(self) -> str | None:
        parts = [self.initial_prompt.strip()] if self.initial_prompt.strip() else []
        if self.hotwords:
            parts.append("、".join(self.hotwords) + "。")
        return " ".join(parts) or None


class VadConfig(BaseModel):
    threshold: float = 0.5
    # Per-source overrides, e.g. {"mac_mic": 0.6, "mac_system": 0.4}: the two tracks sit at
    # very different levels, so one threshold over-triggers on the quiet one.
    threshold_by_source: dict[str, float] = Field(default_factory=dict)
    min_speech_ms: int = 250
    min_silence_ms: int = 500
    # Neighbouring speech regions closer than this are transcribed as one chunk so the
    # model sees whole sentences instead of 1-2 s fragments.
    merge_gap_ms: int = 2000
    max_speech_s: float = 30.0
    pad_ms: int = 200
    # Whisper pads every chunk to its fixed max-window length regardless of how much of it
    # is speech, so a 3 s region costs the same compute as a 30 s one. `merge_gap_ms` runs
    # first and produces the (already-merged, still short) regions this packs; packing then
    # greedily joins consecutive regions into contiguous spans up to `max_speech_s`,
    # absorbing silence gaps up to this value between them so one ASR call amortizes the
    # window padding across several regions instead of paying it once per region. Kept
    # separate from `merge_gap_ms` (which decides what counts as "one utterance") because
    # a much larger gap is fine to absorb here purely to fill the padded window, at the
    # cost of feeding the decoder more silence (which risks repetition/hallucination on
    # long silent stretches, hence the cap rather than absorbing unboundedly). 0 disables
    # packing.
    pack_max_gap_ms: int = 5000

    def for_source(self, source: str) -> VadConfig:
        """This config with `threshold` replaced by the override for `source`, if any."""
        override = self.threshold_by_source.get(source)
        return self if override is None else self.model_copy(update={"threshold": override})


class EpisodesConfig(BaseModel):
    gap_minutes: float = 5.0


class SummarizeConfig(BaseModel):
    map_model: str = "claude-haiku-4-5"
    reduce_model: str = "claude-opus-5"
    timezone: str = "Asia/Tokyo"


class CorrectConfig(BaseModel):
    """Claude-based correction pass over ASR text, before summarization."""

    enabled: bool = True
    model: str = "claude-haiku-4-5"
    # Characters per correction call; longer days are split into consecutive batches.
    batch_chars: int = 6000
    # Utterances the decoder was this unsure of are flagged for the model to look at
    # harder. Whisper's avg_logprob runs about -0.1 (confident) to -1.0 (guessing).
    low_confidence_logprob: float = -0.6


class DeliverConfig(BaseModel):
    slack: bool = False
    email: bool = False
    email_from: str = ""
    email_to: str = ""
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: str = ""
    # Local-only: a macOS notification that the digest is ready. Nothing is sent anywhere.
    notify: bool = True
    repo: bool = False
    # Local clone of a *private* repo; digests contain other people's speech and client names.
    repo_path: str = ""
    repo_subdir: str = "digests"
    repo_branch: str = ""  # empty: use whatever branch is currently checked out
    repo_remote: str = "origin"


class ScheduleConfig(BaseModel):
    digest_hour: int = 22
    digest_minute: int = 0
    worker_poll_seconds: int = 30
    # Wall-clock budget (seconds) `vas digest` may spend transcribing a pending backlog
    # before giving up and summarizing whatever it has. Keeps a dead worker from turning
    # the nightly digest into an hours-long transcription run. 0 disables catch-up.
    digest_catchup_budget_s: int = 900
    # Worker backlog alert: fire one macOS notification when the pending-recording count
    # has stayed at/above `backlog_alert_count` for at least `backlog_alert_minutes`
    # without dropping back below it. 0 for either disables the alert.
    backlog_alert_count: int = 10
    backlog_alert_minutes: int = 30


class LlmConfig(BaseModel):
    # Hard stop: once today's recorded API spend exceeds this, vas refuses further calls.
    daily_budget_usd: float = 2.0


class Config(BaseModel):
    llm: LlmConfig = Field(default_factory=LlmConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    asr: AsrConfig = Field(default_factory=AsrConfig)
    vad: VadConfig = Field(default_factory=VadConfig)
    episodes: EpisodesConfig = Field(default_factory=EpisodesConfig)
    summarize: SummarizeConfig = Field(default_factory=SummarizeConfig)
    correct: CorrectConfig = Field(default_factory=CorrectConfig)
    deliver: DeliverConfig = Field(default_factory=DeliverConfig)
    schedule: ScheduleConfig = Field(default_factory=ScheduleConfig)

    # --- secrets: environment only ---
    @property
    def slack_webhook_url(self) -> str | None:
        return os.environ.get("VAS_SLACK_WEBHOOK_URL")

    @property
    def smtp_password(self) -> str | None:
        return os.environ.get("VAS_SMTP_PASSWORD")

    @property
    def slack_user_token(self) -> str | None:
        """User token (`xoxp-…`, scope `search:read`) for building the glossary from Slack."""
        return os.environ.get("VAS_SLACK_USER_TOKEN")

    def ensure_dirs(self) -> None:
        dirs = [self.paths.inbox, self.paths.store, self.paths.digests]
        if (mirror := self.paths.digest_mirror) is not None:
            dirs.append(mirror)
        for p in dirs:
            p.mkdir(parents=True, exist_ok=True)


def load_config(path: Path | None = None) -> Config:
    """Load config from `path`, `$VAS_CONFIG`, or the default location.

    Missing file → defaults. `$VAS_DATA_DIR` and `$VAS_ASR_MODEL` override the file,
    which keeps tests and one-off runs free of config files.
    """
    candidate = path or Path(os.environ.get("VAS_CONFIG", str(DEFAULT_CONFIG_PATH)))
    candidate = candidate.expanduser()
    data: dict = {}
    if candidate.is_file():
        with candidate.open("rb") as f:
            data = tomllib.load(f)
    cfg = Config.model_validate(data)
    if d := os.environ.get("VAS_DATA_DIR"):
        cfg.paths.data_dir = Path(d)
    if m := os.environ.get("VAS_ASR_MODEL"):
        cfg.asr.model = m
    return cfg
