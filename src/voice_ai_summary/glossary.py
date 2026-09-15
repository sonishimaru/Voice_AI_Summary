"""User glossary: proper nouns and notation conventions.

Feeds three places: ASR hotwords (`asr.py`), the correction pass's prompt
(`correct.py`), and the summarization prompts (`summarize.py`). Built manually
(`vas vocab add`) or extracted from the user's own Slack messages
(`vas vocab import-slack`, via `slack_import.py` + `extract_glossary` below).

Stored as JSON at `<data_dir>/glossary.json`. Only `_call_extract` touches the
network; everything else here is pure and safe to unit test without an API key.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import anthropic
from pydantic import BaseModel, Field

from .config import Config

log = logging.getLogger(__name__)

GLOSSARY_FILENAME = "glossary.json"

# Keep a single extraction call's input well under the model's context window;
# chunk longer message batches and merge the per-chunk extractions.
MAX_EXTRACT_CHARS = 30_000


class Term(BaseModel):
    term: str
    aliases: list[str] = Field(default_factory=list)
    note: str = ""


class Glossary(BaseModel):
    terms: list[Term] = Field(default_factory=list)
    style_notes: list[str] = Field(default_factory=list)

    def hotwords(self) -> list[str]:
        """Terms + aliases, deduplicated, in the order they first appear."""
        seen: set[str] = set()
        out: list[str] = []
        for term in self.terms:
            for word in (term.term, *term.aliases):
                if word and word not in seen:
                    seen.add(word)
                    out.append(word)
        return out

    def merge(self, other: Glossary) -> Glossary:
        """Union by `term`: aliases are unioned, `note` keeps the first non-empty one,
        and `style_notes` are deduplicated. Order is first-seen, `self` before `other`.
        """
        order: list[str] = []
        aliases_by_term: dict[str, list[str]] = {}
        note_by_term: dict[str, str] = {}
        for term in (*self.terms, *other.terms):
            if term.term not in aliases_by_term:
                order.append(term.term)
                aliases_by_term[term.term] = []
                note_by_term[term.term] = ""
            aliases = aliases_by_term[term.term]
            for alias in term.aliases:
                if alias not in aliases:
                    aliases.append(alias)
            if not note_by_term[term.term] and term.note:
                note_by_term[term.term] = term.note

        merged_terms = [
            Term(term=t, aliases=aliases_by_term[t], note=note_by_term[t]) for t in order
        ]

        style_notes: list[str] = []
        for note in (*self.style_notes, *other.style_notes):
            if note not in style_notes:
                style_notes.append(note)

        return Glossary(terms=merged_terms, style_notes=style_notes)

    def prompt_block(self) -> str:
        """Compact Japanese block for prompts. Empty sections are omitted; `""` when
        the whole glossary is empty."""
        parts: list[str] = []
        if self.terms:
            lines = ["## 用語集"]
            for term in self.terms:
                alias_part = f" ({', '.join(term.aliases)})" if term.aliases else ""
                note_part = f": {term.note}" if term.note else ""
                lines.append(f"- {term.term}{alias_part}{note_part}")
            parts.append("\n".join(lines))
        if self.style_notes:
            lines = ["## 表記ルール", *[f"- {note}" for note in self.style_notes]]
            parts.append("\n".join(lines))
        return "\n".join(parts)


def _glossary_path(cfg: Config) -> Path:
    return cfg.paths.root / GLOSSARY_FILENAME


def load_glossary(cfg: Config) -> Glossary:
    """Load the glossary from `<data_dir>/glossary.json`; a missing file is an empty glossary."""
    path = _glossary_path(cfg)
    if not path.is_file():
        return Glossary()
    return Glossary.model_validate_json(path.read_text(encoding="utf-8"))


def save_glossary(cfg: Config, glossary: Glossary) -> None:
    """Write the glossary to `<data_dir>/glossary.json`, creating the directory if needed."""
    path = _glossary_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(glossary.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8"
    )


_EXTRACT_SYSTEM = """あなたはユーザー本人のSlackメッセージから個人用語集を作るアシスタントです。
入力はユーザー自身が投稿したSlackメッセージの一部(1行1メッセージ)です。

以下を抽出してください:
- 固有名詞: 人名(敬称の付け方も含む)、社名、製品名、プロジェクト名、社内用語・略語
  - 読み方や別表記があれば aliases に入れてください
  - どんな人・ものかが分かる短い note を付けてください
- ユーザー自身の表記の流儀(人名の書き方、カタカナ/英語表記の使い分けなど)を
  style_notes に短い文で入れてください

推測で内容を作らず、メッセージ本文に根拠がある項目だけを挙げてください。
すでに分かっている用語集(重複させないための参考。ここにある用語は出力しなくてよい):
{existing}
"""


def _call_extract(
    client: anthropic.Anthropic, model: str, text: str, existing_block: str
) -> Glossary:
    """Network call: extract glossary terms/style notes from one batch of Slack messages.

    Structured outputs guarantee schema-valid JSON, so no assistant prefill is used.
    """
    system = _EXTRACT_SYSTEM.replace("{existing}", existing_block or "(なし)")
    try:
        response = client.messages.parse(
            model=model,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": text}],
            output_format=Glossary,
        )
    except anthropic.RateLimitError:
        log.error("rate limited calling glossary extraction model %s", model)
        raise
    except anthropic.APIConnectionError:
        log.error("network error calling glossary extraction model %s", model)
        raise
    except anthropic.APIStatusError as e:
        log.error("glossary extraction model %s returned status %s", model, e.status_code)
        raise
    return response.parsed_output


def _chunk_messages(messages: list[str], max_chars: int) -> list[str]:
    """Join non-empty `messages` (one per line) into batches of at most `max_chars`."""
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for msg in messages:
        text = msg.strip()
        if not text:
            continue
        line_len = len(text) + 1
        if current and size + line_len > max_chars:
            chunks.append("\n".join(current))
            current, size = [], 0
        current.append(text)
        size += line_len
    if current:
        chunks.append("\n".join(current))
    return chunks


def extract_glossary(
    client: anthropic.Anthropic, model: str, messages: list[str], existing: Glossary
) -> Glossary:
    """Extract glossary terms/style notes from Slack `messages`, chunked to stay under the
    model's context window, merging each chunk's extraction into `existing` progressively."""
    result = existing
    for chunk in _chunk_messages(messages, MAX_EXTRACT_CHARS):
        extracted = _call_extract(client, model, chunk, result.prompt_block())
        result = result.merge(extracted)
    return result
