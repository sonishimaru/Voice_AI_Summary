"""Tests for Claude-based summarization. Network calls are stubbed by monkeypatching
`_call_map`/`_call_reduce` directly - no anthropic client ever touches the network."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from voice_ai_summary import summarize as summarize_mod
from voice_ai_summary.config import Config
from voice_ai_summary.summarize import (
    PROMPT_VERSION,
    EpisodeSummary,
    run_day,
)


def _insert_recording(
    conn: sqlite3.Connection, *, source: str, started_at_utc: str, sha256: str
) -> int:
    cur = conn.execute(
        """
        INSERT INTO recordings(
            source, device_id, started_at_utc, tz_offset, duration_ms,
            sha256, storage_path, original_name, ingested_at, processed_at
        ) VALUES (?, 'dev1', ?, '+00:00', 1000, ?, 'store/x.wav', 'x.wav', ?, ?)
        """,
        (source, started_at_utc, sha256, started_at_utc, started_at_utc),
    )
    return cur.lastrowid


def _insert_utterance(
    conn: sqlite3.Connection,
    *,
    recording_id: int,
    rec_started_at_utc: str,
    t_start_ms: int,
    t_end_ms: int,
    speaker: str,
    text: str = "テスト発話",
) -> int:
    base = datetime.strptime(rec_started_at_utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    abs_start = (base + timedelta(milliseconds=t_start_ms)).strftime("%Y-%m-%dT%H:%M:%SZ")
    cur = conn.execute(
        """
        INSERT INTO utterances(
            recording_id, t_start_ms, t_end_ms, abs_start_utc, text, lang,
            asr_model, avg_logprob, speaker
        ) VALUES (?, ?, ?, ?, ?, 'ja', 'fake', -0.1, ?)
        """,
        (recording_id, t_start_ms, t_end_ms, abs_start, text, speaker),
    )
    return cur.lastrowid


def _seed_one_episode(conn: sqlite3.Connection, day: str = "2026-09-15") -> None:
    started = f"{day}T00:00:00Z"
    rec = _insert_recording(conn, source="mac_mic", started_at_utc=started, sha256="e" * 64)
    conn.commit()
    for i in range(3):
        _insert_utterance(
            conn,
            recording_id=rec,
            rec_started_at_utc=started,
            t_start_ms=i * 1_000,
            t_end_ms=i * 1_000 + 500,
            speaker="me",
        )
    conn.commit()


_CANNED_EPISODE_SUMMARY = EpisodeSummary(
    title="テストエピソード",
    kind_guess="solo",
    topics=["トピックA"],
    decisions=["決定A"],
    action_items=[],
    people=[],
    open_questions=["未解決A"],
    notable_quotes=[],
    summary_ja="これはテスト要約です。",
)
_CANNED_DAY_MARKDOWN = "# 2026-09-15 の記録\n\n## ハイライト\n\nテスト\n"


@pytest.fixture
def fake_calls(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    counts = {"map": 0, "reduce": 0}

    def fake_map(client, model, transcript, meta):
        counts["map"] += 1
        return _CANNED_EPISODE_SUMMARY

    def fake_reduce(client, model, day, episodes_json):
        counts["reduce"] += 1
        return _CANNED_DAY_MARKDOWN

    monkeypatch.setattr(summarize_mod, "_call_map", fake_map)
    monkeypatch.setattr(summarize_mod, "_call_reduce", fake_reduce)
    return counts


def test_run_day_stores_summaries_and_writes_digest(vas, fake_calls) -> None:
    cfg: Config
    cfg, conn = vas
    _seed_one_episode(conn)

    markdown = run_day(conn, cfg, "2026-09-15", client=object())

    assert markdown == _CANNED_DAY_MARKDOWN
    assert fake_calls["map"] == 1
    assert fake_calls["reduce"] == 1

    ep_row = conn.execute("SELECT * FROM summaries WHERE scope='episode'").fetchone()
    assert ep_row is not None
    assert ep_row["prompt_version"] == PROMPT_VERSION
    stored = EpisodeSummary.model_validate_json(ep_row["json"])
    assert stored.title == "テストエピソード"

    day_row = conn.execute("SELECT * FROM summaries WHERE scope='day'").fetchone()
    assert day_row is not None
    assert day_row["scope_key"] == "2026-09-15"
    assert day_row["markdown"] == _CANNED_DAY_MARKDOWN

    episode_title = conn.execute("SELECT title FROM episodes").fetchone()["title"]
    assert episode_title == "テストエピソード"

    digest_path = cfg.paths.digests / "2026-09-15.md"
    assert digest_path.is_file()
    assert digest_path.read_text(encoding="utf-8") == _CANNED_DAY_MARKDOWN


def test_run_day_second_call_makes_zero_api_calls(vas, fake_calls) -> None:
    cfg, conn = vas
    _seed_one_episode(conn)

    run_day(conn, cfg, "2026-09-15", client=object())
    assert fake_calls["map"] == 1
    assert fake_calls["reduce"] == 1

    markdown = run_day(conn, cfg, "2026-09-15", client=object())

    assert markdown == _CANNED_DAY_MARKDOWN
    assert fake_calls["map"] == 1
    assert fake_calls["reduce"] == 1


def test_run_day_force_recalls(vas, fake_calls) -> None:
    cfg, conn = vas
    _seed_one_episode(conn)

    run_day(conn, cfg, "2026-09-15", client=object())
    assert fake_calls["map"] == 1
    assert fake_calls["reduce"] == 1

    run_day(conn, cfg, "2026-09-15", client=object(), force=True)
    assert fake_calls["map"] == 2
    assert fake_calls["reduce"] == 2


def test_run_day_empty_day_skips_api_and_writes_no_data_markdown(vas, fake_calls) -> None:
    cfg, conn = vas

    markdown = run_day(conn, cfg, "2026-09-16")

    assert "記録なし" in markdown
    assert fake_calls["map"] == 0
    assert fake_calls["reduce"] == 0

    digest_path = cfg.paths.digests / "2026-09-16.md"
    assert digest_path.is_file()
    assert digest_path.read_text(encoding="utf-8") == markdown

    day_row = conn.execute(
        "SELECT * FROM summaries WHERE scope='day' AND scope_key='2026-09-16'"
    ).fetchone()
    assert day_row is not None
    assert day_row["markdown"] == markdown


def test_run_day_recomputes_when_new_utterances_arrive(vas, fake_calls, monkeypatch) -> None:
    """Late-processed audio changes an episode's transcript → the map step runs again; the
    day rollup is re-run only when the episode summaries it consumes actually changed."""
    cfg, conn = vas
    _seed_one_episode(conn)
    run_day(conn, cfg, "2026-09-15", client=object())
    assert fake_calls == {"map": 1, "reduce": 1}

    def add_utterance(t_start_ms: int, text: str) -> None:
        rec = conn.execute("SELECT id, started_at_utc FROM recordings").fetchone()
        _insert_utterance(
            conn,
            recording_id=rec["id"],
            rec_started_at_utc=rec["started_at_utc"],
            t_start_ms=t_start_ms,
            t_end_ms=t_start_ms + 500,
            speaker="me",
            text=text,
        )
        conn.commit()

    # Transcript changed but the (canned) episode summary is identical → no reduce call.
    add_utterance(4_000, "追加の発話")
    run_day(conn, cfg, "2026-09-15", client=object())
    assert fake_calls == {"map": 2, "reduce": 1}
    # Episode ids were reused by the rebuild, but the title is still populated.
    assert conn.execute("SELECT title FROM episodes").fetchone()["title"] == "テストエピソード"

    # Transcript changed and the episode summary differs → reduce runs again.
    changed = _CANNED_EPISODE_SUMMARY.model_copy(update={"title": "新しいタイトル"})

    def fake_map_changed(client, model, transcript, meta):
        fake_calls["map"] += 1
        return changed

    monkeypatch.setattr(summarize_mod, "_call_map", fake_map_changed)
    add_utterance(5_000, "さらに追加")
    run_day(conn, cfg, "2026-09-15", client=object())
    assert fake_calls == {"map": 3, "reduce": 2}
    assert conn.execute("SELECT title FROM episodes").fetchone()["title"] == "新しいタイトル"
