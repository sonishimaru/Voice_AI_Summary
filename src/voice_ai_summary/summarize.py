"""Claude-based summarization: per-episode extraction (map) + daily rollup (reduce).

Exactly two functions in this module touch the network - `_call_map` and
`_call_reduce`. Everything else is pure and safe to unit test without an API key by
injecting a fake `client` (or by monkeypatching `_call_map`/`_call_reduce`).
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3

import anthropic
from pydantic import BaseModel, Field

from .config import Config
from .db import utcnow_iso
from .episodes import build_episodes, episode_transcript
from .timeutil import fmt_hm

log = logging.getLogger(__name__)

PROMPT_VERSION = "v1"

# Keep a single map call's transcript well under the model's context window; chunk
# longer episodes and merge the per-chunk extractions.
MAX_TRANSCRIPT_CHARS = 60_000
MAX_CHUNK_CHARS = 40_000


class ActionItem(BaseModel):
    text: str
    owner: str | None = None
    due: str | None = None


class Quote(BaseModel):
    text: str
    speaker: str


class EpisodeSummary(BaseModel):
    title: str
    kind_guess: str
    topics: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    people: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    notable_quotes: list[Quote] = Field(default_factory=list)
    summary_ja: str = ""


class EpisodeWithRange(BaseModel):
    """One episode's extraction plus the local time range it covers - the shape fed
    into the daily reduce step."""

    episode_id: int
    start: str
    end: str
    kind: str
    summary: EpisodeSummary


DailySummaryInput = list[EpisodeWithRange]


_MAP_SYSTEM = """あなたはユーザー本人の一日の音声ログを整理するアシスタントです。
入力はローカルの音声認識(ASR)で生成された、あるユーザーの1つの「エピソード」
(ひとつながりの活動)の文字起こしです。
話者ラベルの意味: [me] はユーザー本人、[other] は相手(電話や会話の相手)、
[unknown] は不明な話者です。

注意点:
- ASRには誤字や欠落がある可能性があります。文脈から明らかな場合のみ補って読み、
  存在しない内容は決して創作しないでください。
- 発言されていない決定事項やタスクを推測で追加しないでください。
  曖昧な場合は open_questions に入れてください。
- 出力は日本語で、簡潔かつ忠実にしてください。
- kind_guess は "call"(電話・通話)、"solo"(独り言・作業)、
  "media"(動画・音声コンテンツの再生)、"ambient"(周辺音・雑談)のいずれかにしてください。
- メタデータの kind が "media" のエピソードは、ユーザーが視聴していた動画・音声
  (ニュース、YouTube、アニメなど)の音声である可能性が高いです。その場合は
  「何を視聴していたか」を短くまとめ、登場人物の発言をユーザーの決定事項や
  TODO として扱わないでください。
"""

_REDUCE_SYSTEM = """あなたはユーザー本人の一日分の音声ログ要約(デイリーダイジェスト)を
作成するアシスタントです。
入力は、その日の各エピソードについて既に抽出された要約データ(JSON配列)です。
生の文字起こしは含まれていません。
このJSONだけを根拠にして、日本語のMarkdownで一日のダイジェストを作成してください。
存在しない情報を創作しないでください。
kind が "media" のエピソードは視聴していたコンテンツなので、エピソード一覧では
タイトルの先頭に【視聴】を付けて 1 行にまとめ、ハイライト・決定事項・TODO には含めないでください。

出力フォーマット(必ずこの見出し構成に従うこと):
# {day} の記録
## ハイライト
## 決定事項
## TODO / アクション
## エピソード一覧
(各エピソードについて: 時間帯、タイトル、2〜3行の要約)
## 未解決の質問
"""


def _chunk_transcript(transcript: str, max_chars: int) -> list[str]:
    """Split `transcript` on line boundaries into pieces no longer than `max_chars`."""
    lines = transcript.split("\n")
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for line in lines:
        line_len = len(line) + 1
        if current and size + line_len > max_chars:
            chunks.append("\n".join(current))
            current = []
            size = 0
        current.append(line)
        size += line_len
    if current:
        chunks.append("\n".join(current))
    return chunks


def _call_map(
    client: anthropic.Anthropic, model: str, transcript: str, meta: dict
) -> EpisodeSummary:
    """Map step: extract a structured `EpisodeSummary` from one episode's transcript.

    Structured outputs guarantee schema-valid JSON, so no assistant prefill is used
    (and would 400 on these models anyway). `claude-haiku-4-5` needs no `thinking` param.
    """
    user_content = (
        f"エピソード情報: {meta.get('start')}〜{meta.get('end')} "
        f"種別(推定)={meta.get('kind')} ソース={meta.get('source_mix')}\n\n"
        f"文字起こし:\n{transcript}"
    )
    try:
        response = client.messages.parse(
            model=model,
            max_tokens=4096,
            system=_MAP_SYSTEM,
            messages=[{"role": "user", "content": user_content}],
            output_format=EpisodeSummary,
        )
    except anthropic.RateLimitError:
        log.error("rate limited calling map model %s", model)
        raise
    except anthropic.APIConnectionError:
        log.error("network error calling map model %s", model)
        raise
    except anthropic.APIStatusError as e:
        log.error("map model %s returned status %s", model, e.status_code)
        raise
    return response.parsed_output


def _merge_str_lists(lists: list[list[str]]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for lst in lists:
        for item in lst:
            if item not in seen:
                seen.add(item)
                out.append(item)
    return out


def _merge_model_lists(lists: list[list[BaseModel]]) -> list[BaseModel]:
    seen: set[str] = set()
    out: list[BaseModel] = []
    for lst in lists:
        for item in lst:
            key = item.model_dump_json()
            if key not in seen:
                seen.add(key)
                out.append(item)
    return out


def _merge_episode_summaries(parts: list[EpisodeSummary]) -> EpisodeSummary:
    if len(parts) == 1:
        return parts[0]
    return EpisodeSummary(
        title=parts[0].title,
        kind_guess=parts[0].kind_guess,
        topics=_merge_str_lists([p.topics for p in parts]),
        decisions=_merge_str_lists([p.decisions for p in parts]),
        action_items=_merge_model_lists([p.action_items for p in parts]),
        people=_merge_str_lists([p.people for p in parts]),
        open_questions=_merge_str_lists([p.open_questions for p in parts]),
        notable_quotes=_merge_model_lists([p.notable_quotes for p in parts]),
        summary_ja="\n".join(p.summary_ja for p in parts if p.summary_ja),
    )


def summarize_episode(
    client: anthropic.Anthropic, cfg: Config, transcript: str, meta: dict
) -> EpisodeSummary:
    """Extract a structured summary for one episode, chunking very long transcripts."""
    model = cfg.summarize.map_model
    if len(transcript) <= MAX_TRANSCRIPT_CHARS:
        return _call_map(client, model, transcript, meta)
    chunks = _chunk_transcript(transcript, MAX_CHUNK_CHARS)
    parts = [_call_map(client, model, chunk, meta) for chunk in chunks]
    return _merge_episode_summaries(parts)


def _call_reduce(client: anthropic.Anthropic, model: str, day: str, episodes_json: str) -> str:
    """Reduce step: turn the day's episode summaries into a Japanese Markdown digest.

    `claude-opus-5` has adaptive thinking on by default, so `thinking` is omitted;
    `output_config.effort` tunes it. Streamed (`.get_final_message()`) since the
    digest can be long. The README's server-side refusal-fallback pattern
    (`betas=[...]` + `fallbacks=...` on `client.beta.messages...`) is documented only
    for `claude-fable-5-1` with an explicit fallback model - it is not shown for
    `claude-opus-5`, so per the task's guidance we use the plain `client.messages`
    call here instead of guessing at an unlisted beta/model pairing.
    """
    user_content = f"{day} のエピソード要約(JSON配列):\n{episodes_json}"
    try:
        with client.messages.stream(
            model=model,
            max_tokens=16000,
            output_config={"effort": "medium"},
            system=_REDUCE_SYSTEM.replace("{day}", day),
            messages=[{"role": "user", "content": user_content}],
        ) as stream:
            message = stream.get_final_message()
    except anthropic.RateLimitError:
        log.error("rate limited calling reduce model %s", model)
        raise
    except anthropic.APIConnectionError:
        log.error("network error calling reduce model %s", model)
        raise
    except anthropic.APIStatusError as e:
        log.error("reduce model %s returned status %s", model, e.status_code)
        raise
    return "".join(block.text for block in message.content if block.type == "text")


def summarize_day(client: anthropic.Anthropic, cfg: Config, day: str, episodes: list[dict]) -> str:
    """Reduce step: build the day's Markdown digest from episode JSONs only (no
    raw transcript is ever sent here)."""
    episodes_json = json.dumps(episodes, ensure_ascii=False, indent=2)
    return _call_reduce(client, cfg.summarize.reduce_model, day, episodes_json)


def _upsert_summary(
    conn: sqlite3.Connection,
    *,
    scope: str,
    scope_key: str,
    model: str,
    json_str: str,
    markdown: str | None,
) -> None:
    conn.execute(
        """
        INSERT INTO summaries(scope, scope_key, model, prompt_version, json, markdown, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(scope, scope_key, prompt_version)
        DO UPDATE SET model = excluded.model,
                      json = excluded.json,
                      markdown = excluded.markdown,
                      created_at = excluded.created_at
        """,
        (scope, scope_key, model, PROMPT_VERSION, json_str, markdown, utcnow_iso()),
    )
    conn.commit()


def _no_data_markdown(day: str) -> str:
    return f"# {day} の記録\n\n記録なし\n"


def _content_key(text: str) -> str:
    """Stable cache key for summaries derived from `text`."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def run_day(
    conn: sqlite3.Connection,
    cfg: Config,
    day: str,
    *,
    client: anthropic.Anthropic | None = None,
    force: bool = False,
) -> str:
    """Build the day's episodes, summarize any that need it, roll up a day digest,
    store both in `summaries`, and write the digest to `<digests>/{day}.md`.

    Episode/day summaries already stored under `PROMPT_VERSION` are reused unless
    `force` is set. A day with no utterances never touches the API.
    """
    tz = cfg.summarize.timezone
    episode_ids = build_episodes(conn, cfg, day)

    cfg.paths.digests.mkdir(parents=True, exist_ok=True)
    digest_path = cfg.paths.digests / f"{day}.md"

    if not episode_ids:
        markdown = _no_data_markdown(day)
        _upsert_summary(
            conn, scope="day", scope_key=day, model="none", json_str="{}", markdown=markdown
        )
        digest_path.write_text(markdown, encoding="utf-8")
        return markdown

    if client is None:
        client = anthropic.Anthropic()

    episode_payloads: list[dict] = []
    for episode_id in episode_ids:
        erow = conn.execute("SELECT * FROM episodes WHERE id = ?", (episode_id,)).fetchone()
        transcript = episode_transcript(conn, episode_id, tz)
        # Episodes are rebuilt (and re-numbered) on every run, so the cache is keyed by
        # transcript content rather than by episode id.
        cache_key = _content_key(transcript)

        row = None
        if not force:
            row = conn.execute(
                "SELECT * FROM summaries"
                " WHERE scope='episode' AND scope_key=? AND prompt_version=?",
                (cache_key, PROMPT_VERSION),
            ).fetchone()

        if row is None:
            meta = {
                "start": fmt_hm(erow["started_at_utc"], tz),
                "end": fmt_hm(erow["ended_at_utc"], tz),
                "kind": erow["kind"],
                "source_mix": erow["source_mix"],
            }
            ep_summary = summarize_episode(client, cfg, transcript, meta)
            _upsert_summary(
                conn,
                scope="episode",
                scope_key=cache_key,
                model=cfg.summarize.map_model,
                json_str=ep_summary.model_dump_json(),
                markdown=None,
            )
        else:
            ep_summary = EpisodeSummary.model_validate_json(row["json"])

        conn.execute("UPDATE episodes SET title = ? WHERE id = ?", (ep_summary.title, episode_id))
        conn.commit()

        episode_payloads.append(
            {
                "episode_id": episode_id,
                "start": fmt_hm(erow["started_at_utc"], tz),
                "end": fmt_hm(erow["ended_at_utc"], tz),
                "kind": erow["kind"],
                "summary": ep_summary.model_dump(),
            }
        )

    # The day rollup is reused only while its inputs are unchanged (late-processed audio
    # can add utterances to a day that was already summarised).
    payload_hash = _content_key(
        json.dumps(
            [{k: v for k, v in e.items() if k != "episode_id"} for e in episode_payloads],
            sort_keys=True,
            ensure_ascii=False,
        )
    )
    day_row = None
    if not force:
        day_row = conn.execute(
            "SELECT * FROM summaries WHERE scope='day' AND scope_key=? AND prompt_version=?",
            (day, PROMPT_VERSION),
        ).fetchone()
        if day_row is not None and json.loads(day_row["json"]).get("content_hash") != payload_hash:
            day_row = None

    if day_row is None:
        markdown = summarize_day(client, cfg, day, episode_payloads)
        _upsert_summary(
            conn,
            scope="day",
            scope_key=day,
            model=cfg.summarize.reduce_model,
            json_str=json.dumps(
                {"content_hash": payload_hash, "episodes": episode_payloads}, ensure_ascii=False
            ),
            markdown=markdown,
        )
    else:
        markdown = day_row["markdown"]

    digest_path.write_text(markdown, encoding="utf-8")
    return markdown
