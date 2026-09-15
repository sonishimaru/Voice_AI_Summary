"""Claude-based correction pass over ASR text.

Local Whisper transcripts have errors in proper nouns and homophones; this module asks
Claude to fix only those, using context and the user's glossary (`glossary.py`) - text
only, audio never leaves the Mac. Exactly one function here touches the network -
`_call_correct` - mirroring `summarize.py::_call_map`; everything else is pure and safe
to unit test without an API key by injecting a fake `client` (or by monkeypatching
`_call_correct`).
"""

from __future__ import annotations

import logging
import sqlite3

import anthropic
import pydantic
from pydantic import BaseModel, Field

from .config import Config
from .db import transaction, utcnow_iso
from .glossary import OutputTruncated, load_glossary
from .timeutil import fmt_hm, local_day_bounds

log = logging.getLogger(__name__)

_CORRECT_SYSTEM = """あなたはユーザー本人の一日の音声ログの文字起こしを校正するアシスタントです。
入力はローカルの音声認識(ASR)で生成された、ある一日の発話を時系列に並べたものです。
各行は "ID<タブ>HH:MM<タブ>[話者]<タブ>本文" の形式(タブ区切り)です。
話者ラベルの意味: [me] はユーザー本人、[other] は相手、[unknown] は不明な話者です。

やること:
- 前後の文脈と(付与されていれば)用語集を参考に、明らかなASRの誤り
  (同音異義語の誤変換、固有名詞の誤字、意味の通らない不自然な断片)だけを修正してください。
- 口調・方言・フィラー(「あの」「えー」など)は誤りではないので変更しないでください。
- 発言されていない内容を追加したり、逆に内容を削除したりしないでください。
- 行を統合・分割しないでください。1行は1発話のままにしてください。
- 修正が必要な行だけを、そのIDと修正後の本文のペアとして返してください。
  修正不要な行は出力に含めないでください。
"""


class Fix(BaseModel):
    id: int
    text: str


class CorrectionResult(BaseModel):
    fixes: list[Fix] = Field(default_factory=list)


def _call_correct(
    client: anthropic.Anthropic, model: str, block: str, glossary_block: str
) -> CorrectionResult:
    """Network call: ask Claude which lines in `block` contain a clear ASR error.

    Structured outputs guarantee schema-valid JSON, so no assistant prefill is used.
    """
    user_content = f"文字起こし:\n{block}"
    if glossary_block:
        user_content += f"\n\n{glossary_block}"
    try:
        response = client.messages.parse(
            model=model,
            max_tokens=16000,
            system=_CORRECT_SYSTEM,
            messages=[{"role": "user", "content": user_content}],
            output_format=CorrectionResult,
        )
    except pydantic.ValidationError as e:
        # Structured output only fails validation when cut off at max_tokens.
        raise OutputTruncated(str(e)) from e
    except anthropic.RateLimitError:
        log.error("rate limited calling correction model %s", model)
        raise
    except anthropic.APIConnectionError:
        log.error("network error calling correction model %s", model)
        raise
    except anthropic.APIStatusError as e:
        log.error("correction model %s returned status %s", model, e.status_code)
        raise
    return response.parsed_output


def _format_line(row: sqlite3.Row, tz: str) -> str:
    return f"{row['id']}\t{fmt_hm(row['abs_start_utc'], tz)}\t[{row['speaker']}]\t{row['text']}"


def _batch_rows(
    rows: list[sqlite3.Row], batch_chars: int, tz: str
) -> list[tuple[list[sqlite3.Row], str]]:
    """Group consecutive `rows` into batches whose rendered block is <= `batch_chars`."""
    batches: list[tuple[list[sqlite3.Row], str]] = []
    current_rows: list[sqlite3.Row] = []
    current_lines: list[str] = []
    size = 0
    for row in rows:
        line = _format_line(row, tz)
        line_len = len(line) + 1
        if current_rows and size + line_len > batch_chars:
            batches.append((current_rows, "\n".join(current_lines)))
            current_rows, current_lines, size = [], [], 0
        current_rows.append(row)
        current_lines.append(line)
        size += line_len
    if current_rows:
        batches.append((current_rows, "\n".join(current_lines)))
    return batches


def correct_day(
    conn: sqlite3.Connection,
    cfg: Config,
    day: str,
    *,
    client: anthropic.Anthropic | None = None,
    force: bool = False,
) -> int:
    """Correct a local day's ASR text in place. Returns the number of utterances changed.

    Selects the day's utterances (joined with `recordings` for `source`, ordered by
    `abs_start_utc, t_start_ms`) where `corrected_at IS NULL`, or all of them if `force`.
    They are grouped into consecutive batches of <= `cfg.correct.batch_chars` characters;
    each batch is one Claude call and one DB transaction. For a fix whose id is in the
    batch and whose text differs and is non-empty: the original text is preserved in
    `raw_text` (only the first time), `text` is replaced, and `corrected_at`/
    `correction_model` are stamped. Unchanged utterances only get `corrected_at`/
    `correction_model` stamped, so they are not re-sent on the next call.

    The API is never touched when there is nothing to correct - `client` (default
    `anthropic.Anthropic()`) is only constructed once there is at least one row to send.
    """
    tz = cfg.summarize.timezone
    start_utc, end_utc = local_day_bounds(day, tz)
    corrected_clause = "" if force else " AND u.corrected_at IS NULL"
    rows = conn.execute(
        f"""
        SELECT u.id AS id, u.abs_start_utc AS abs_start_utc, u.speaker AS speaker,
               u.text AS text, r.source AS source
        FROM utterances u
        JOIN recordings r ON r.id = u.recording_id
        WHERE u.abs_start_utc >= ? AND u.abs_start_utc < ?{corrected_clause}
        ORDER BY u.abs_start_utc, u.t_start_ms
        """,
        (start_utc, end_utc),
    ).fetchall()

    if not rows:
        return 0

    glossary_block = load_glossary(cfg).prompt_block()
    batches = _batch_rows(rows, cfg.correct.batch_chars, tz)

    if client is None:
        client = anthropic.Anthropic()

    changed = 0
    for batch_rows, block_text in batches:
        changed += _correct_batch(conn, cfg, client, batch_rows, block_text, glossary_block, tz)
    return changed


def _correct_batch(
    conn: sqlite3.Connection,
    cfg: Config,
    client: anthropic.Anthropic,
    batch_rows: list[sqlite3.Row],
    block_text: str,
    glossary_block: str,
    tz: str,
) -> int:
    """Correct one batch; on a truncated response, split the batch in half and retry."""
    try:
        result = _call_correct(client, cfg.correct.model, block_text, glossary_block)
    except OutputTruncated:
        if len(batch_rows) < 2:
            raise
        log.warning("correction output truncated; splitting batch of %d rows", len(batch_rows))
        mid = len(batch_rows) // 2
        total = 0
        for part in (batch_rows[:mid], batch_rows[mid:]):
            [(rows, block)] = _batch_rows(part, 10**9, tz)
            total += _correct_batch(conn, cfg, client, rows, block, glossary_block, tz)
        return total
    fixes = {fix.id: fix.text for fix in result.fixes}
    now = utcnow_iso()
    changed = 0
    with transaction(conn):
        for row in batch_rows:
            new_text = fixes.get(row["id"])
            if new_text is not None and new_text.strip() and new_text != row["text"]:
                conn.execute(
                    "UPDATE utterances SET raw_text = COALESCE(raw_text, text), "
                    "text = ?, corrected_at = ?, correction_model = ? WHERE id = ?",
                    (new_text, now, cfg.correct.model, row["id"]),
                )
                changed += 1
            else:
                conn.execute(
                    "UPDATE utterances SET corrected_at = ?, correction_model = ? WHERE id = ?",
                    (now, cfg.correct.model, row["id"]),
                )
    return changed
