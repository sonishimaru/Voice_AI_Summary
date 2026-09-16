"""Configuration: TOML file + environment variables (secrets are env-only)."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

from pydantic import BaseModel, Field

DEFAULT_CONFIG_PATH = Path("~/.config/voice-ai-summary/config.toml")


class PathsConfig(BaseModel):
    data_dir: Path = Path("~/Library/Application Support/VoiceAISummary")

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


DEFAULT_FASTER_WHISPER_MODEL = "kotoba-tech/kotoba-whisper-v2.0-faster"
DEFAULT_MLX_MODEL = "mlx-community/whisper-large-v3-turbo"


class AsrConfig(BaseModel):
    backend: str = "faster-whisper"  # faster-whisper | mlx
    model: str = DEFAULT_FASTER_WHISPER_MODEL
    device: str = "auto"
    compute_type: str = "default"
    language: str = "ja"
    beam_size: int = 5
    # Domain vocabulary (names, products, jargon) biases decoding toward these spellings.
    hotwords: list[str] = Field(default_factory=list)
    initial_prompt: str = ""

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
    min_speech_ms: int = 250
    min_silence_ms: int = 500
    # Neighbouring speech regions closer than this are transcribed as one chunk so the
    # model sees whole sentences instead of 1-2 s fragments.
    merge_gap_ms: int = 2000
    max_speech_s: float = 30.0
    pad_ms: int = 200


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


class DeliverConfig(BaseModel):
    slack: bool = False
    email: bool = False
    email_from: str = ""
    email_to: str = ""
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: str = ""
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
        for p in (self.paths.inbox, self.paths.store, self.paths.digests):
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
