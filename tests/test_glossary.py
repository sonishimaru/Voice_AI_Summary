"""Tests for the personal glossary: storage, merge/format logic, Slack import, and
extraction. No network calls - `import_from_slack` uses `httpx.MockTransport` and
`extract_glossary` is tested with a monkeypatched `_call_extract`."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from voice_ai_summary import glossary as glossary_mod
from voice_ai_summary.config import Config
from voice_ai_summary.glossary import (
    Glossary,
    Term,
    extract_glossary,
    load_glossary,
    save_glossary,
)
from voice_ai_summary.slack_import import import_from_slack


def _cfg(tmp_path) -> Config:
    cfg = Config()
    cfg.paths.data_dir = tmp_path
    return cfg


def test_load_missing_glossary_is_empty(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    glossary = load_glossary(cfg)
    assert glossary == Glossary()
    assert glossary.prompt_block() == ""


def test_save_and_load_roundtrip(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    original = Glossary(
        terms=[Term(term="西丸", aliases=["にしまる"], note="ユーザー本人の姓")],
        style_notes=["社名は「チョコレート」と表記"],
    )
    save_glossary(cfg, original)
    loaded = load_glossary(cfg)
    assert loaded == original
    assert (cfg.paths.root / "glossary.json").is_file()


def test_hotwords_dedupe_and_order() -> None:
    glossary = Glossary(
        terms=[
            Term(term="西丸", aliases=["にしまる", "西丸"]),
            Term(term="安田さん", aliases=["西丸"]),
        ]
    )
    assert glossary.hotwords() == ["西丸", "にしまる", "安田さん"]


def test_merge_unions_terms_aliases_and_style_notes() -> None:
    a = Glossary(
        terms=[Term(term="西丸", aliases=["にしまる"], note="ユーザー本人の姓")],
        style_notes=["社名は「チョコレート」と表記"],
    )
    b = Glossary(
        terms=[
            Term(term="西丸", aliases=["ニシマル"]),
            Term(term="安田さん", note="取引先の担当者"),
        ],
        style_notes=["社名は「チョコレート」と表記", "英字は半角"],
    )

    merged = a.merge(b)

    assert [t.term for t in merged.terms] == ["西丸", "安田さん"]
    nishimaru = next(t for t in merged.terms if t.term == "西丸")
    assert nishimaru.aliases == ["にしまる", "ニシマル"]
    assert nishimaru.note == "ユーザー本人の姓"
    yasuda = next(t for t in merged.terms if t.term == "安田さん")
    assert yasuda.note == "取引先の担当者"
    assert merged.style_notes == ["社名は「チョコレート」と表記", "英字は半角"]


def test_merge_folds_alias_spellings_of_the_same_entity() -> None:
    """Chunked extraction names the same thing twice, abbreviation-first in one chunk
    and spelled-out in the next; both spellings land on one entry."""
    a = Glossary(terms=[Term(term="KT", aliases=["KILLTUBE"], note="キルチューブ企画")])
    b = Glossary(
        terms=[
            Term(term="KILLTUBE", aliases=["KT"], note="ユーザーの企画"),
            Term(term="松浦藍さん", aliases=["松浦さん"], note="動画の監督"),
        ]
    )

    merged = a.merge(b)

    assert [t.term for t in merged.terms] == ["KT", "松浦藍さん"]
    assert merged.terms[0].aliases == ["KILLTUBE"]
    assert merged.terms[0].note == "キルチューブ企画"


def test_merge_keeps_an_entity_that_is_only_claimed_as_an_alias() -> None:
    """A character listed as an alias of its project is still its own entry - but a bare
    repeat of a known alias, with nothing to add, is folded in."""
    project = Glossary(
        terms=[Term(term="ホイップラピッド", aliases=["ホイラピ", "ホイッピ"], note="キャラ企画")]
    )
    later = Glossary(
        terms=[
            Term(term="ホイッピ", note="ホイップラピッドのキャラクター"),
            Term(term="ホイラピ"),
        ]
    )

    merged = project.merge(later)

    assert [t.term for t in merged.terms] == ["ホイップラピッド", "ホイッピ"]
    assert merged.terms[0].aliases == ["ホイラピ", "ホイッピ"]


def test_merge_matches_surface_forms_ignoring_case_width_and_spacing() -> None:
    a = Glossary(terms=[Term(term="King of Time", note="勤怠管理")])
    b = Glossary(terms=[Term(term="KingOfTime", aliases=["ｷﾝｸﾞｵﾌﾞﾀｲﾑ"], note="勤怠")])

    merged = a.merge(b)

    assert [t.term for t in merged.terms] == ["King of Time"]
    assert merged.terms[0].aliases == ["ｷﾝｸﾞｵﾌﾞﾀｲﾑ"]


def test_merge_collapses_restatements_of_a_style_note() -> None:
    a = Glossary(
        style_notes=["プロジェクト名は英語/カタカナ混在で使用", "キャラクター名は敬称なし"]
    )
    b = Glossary(
        style_notes=[
            "プロジェクト名は英語とカタカナ混在で使用",
            "キャラクター名は敬称なし（ドット、ニコ等）",
            "予算レベルは松竹梅で階級化",
        ]
    )

    merged = a.merge(b)

    assert merged.style_notes == [
        "プロジェクト名は英語/カタカナ混在で使用",
        # The later phrasing spells the same rule out further, so it wins the slot.
        "キャラクター名は敬称なし（ドット、ニコ等）",
        "予算レベルは松竹梅で階級化",
    ]


def test_prompt_block_formatting_and_emptiness() -> None:
    assert Glossary().prompt_block() == ""

    terms_only = Glossary(
        terms=[
            Term(term="西丸", aliases=["にしまる"], note="ユーザー本人の姓"),
            Term(term="安田さん", note="取引先の担当者"),
        ]
    )
    assert terms_only.prompt_block() == (
        "## 用語集\n- 西丸 (にしまる): ユーザー本人の姓\n- 安田さん: 取引先の担当者"
    )

    style_only = Glossary(style_notes=["社名は「チョコレート」と表記"])
    assert style_only.prompt_block() == "## 表記ルール\n- 社名は「チョコレート」と表記"

    both = Glossary(
        terms=[Term(term="西丸", aliases=["にしまる"], note="ユーザー本人の姓")],
        style_notes=["社名は「チョコレート」と表記"],
    )
    assert both.prompt_block() == (
        "## 用語集\n"
        "- 西丸 (にしまる): ユーザー本人の姓\n"
        "## 表記ルール\n"
        "- 社名は「チョコレート」と表記"
    )


def _slack_handler(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/api/auth.test":
        assert request.headers["authorization"] == "Bearer xoxp-test"
        return httpx.Response(200, json={"ok": True, "user_id": "U123"})
    if request.url.path == "/api/search.messages":
        page = int(request.url.params.get("page", "1"))
        expected_after = (datetime.now(UTC) - timedelta(days=90)).strftime("%Y-%m-%d")
        assert request.url.params["query"] == f"from:<@U123> after:{expected_after}"
        if page == 1:
            return httpx.Response(
                200,
                json={
                    "ok": True,
                    "messages": {
                        "matches": [{"text": "メッセージ1"}, {"text": "メッセージ2"}],
                        "pagination": {"page_count": 2},
                    },
                },
            )
        return httpx.Response(
            200,
            json={
                "ok": True,
                "messages": {
                    "matches": [{"text": "メッセージ3"}],
                    "pagination": {"page_count": 2},
                },
            },
        )
    raise AssertionError(f"unexpected request: {request.url}")


def test_import_from_slack_paginates(monkeypatch) -> None:
    monkeypatch.setattr(
        glossary_mod,
        "_call_extract",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not be called")),
    )
    transport = httpx.MockTransport(_slack_handler)
    with httpx.Client(transport=transport) as client_http:
        messages = import_from_slack(client_http, "xoxp-test", days=90, limit=1500)

    assert messages == ["メッセージ1", "メッセージ2", "メッセージ3"]


def test_import_from_slack_raises_on_auth_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": False, "error": "invalid_auth"})

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client_http:
        with pytest.raises(RuntimeError, match="invalid_auth"):
            import_from_slack(client_http, "xoxp-bad")


def test_import_from_slack_stops_at_limit() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/auth.test":
            return httpx.Response(200, json={"ok": True, "user_id": "U123"})
        return httpx.Response(
            200,
            json={
                "ok": True,
                "messages": {
                    "matches": [{"text": f"msg{i}"} for i in range(100)],
                    "pagination": {"page_count": 5},
                },
            },
        )

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client_http:
        messages = import_from_slack(client_http, "xoxp-test", limit=150)

    assert len(messages) == 150


def test_extract_glossary_chunks_and_merges(monkeypatch) -> None:
    long_messages = ["メッセージ" + str(i) * 400 for i in range(60)]  # forces >=2 chunks
    calls: list[str] = []

    def fake_call_extract(client, model, text, existing_block, **kwargs) -> Glossary:
        calls.append(text)
        return Glossary(terms=[Term(term=f"用語{len(calls)}")])

    monkeypatch.setattr(glossary_mod, "_call_extract", fake_call_extract)

    existing = Glossary(terms=[Term(term="既存用語", note="既存")])
    result = extract_glossary(
        client=object(), model="claude-haiku-4-5", messages=long_messages, existing=existing
    )

    assert len(calls) >= 2
    term_names = {t.term for t in result.terms}
    assert "既存用語" in term_names
    for i in range(1, len(calls) + 1):
        assert f"用語{i}" in term_names


def test_extract_glossary_splits_chunk_when_output_truncated(monkeypatch) -> None:
    from voice_ai_summary.glossary import OutputTruncated

    calls: list[int] = []

    def fake_call_extract(client, model, text, existing_block, **kwargs) -> Glossary:
        lines = text.split("\n")
        calls.append(len(lines))
        if len(lines) > 2:
            raise OutputTruncated("cut off")
        return Glossary(terms=[Term(term=lines[0])])

    monkeypatch.setattr(glossary_mod, "_call_extract", fake_call_extract)
    result = extract_glossary(object(), "m", ["a", "b", "c", "d", "e"], Glossary())

    assert calls[0] == 5 and max(calls[1:]) <= 3
    assert [t.term for t in result.terms] == ["a", "c", "d"]


def _channel_handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    q = dict(request.url.params)
    if path == "/api/users.conversations":
        assert q["types"] == "public_channel,private_channel"
        if q.get("cursor") == "c2":
            return httpx.Response(
                200, json={"ok": True, "channels": [{"id": "C2", "name": "design"}]}
            )
        return httpx.Response(
            200,
            json={
                "ok": True,
                "channels": [{"id": "C1", "name": "general"}],
                "response_metadata": {"next_cursor": "c2"},
            },
        )
    if path == "/api/conversations.history":
        assert float(q["oldest"]) > 0
        if q["channel"] == "C1":
            if q.get("cursor") == "h2":
                return httpx.Response(
                    200,
                    json={"ok": True, "messages": [{"text": "般務連絡です"}], "has_more": False},
                )
            return httpx.Response(
                200,
                json={
                    "ok": True,
                    "has_more": True,
                    "response_metadata": {"next_cursor": "h2"},
                    "messages": [
                        {"text": "安田さんに見積もりを送りました"},
                        {"subtype": "channel_join", "text": "joined"},
                        {"bot_id": "B1", "text": "bot noise"},
                        {"text": "   "},
                    ],
                },
            )
        return httpx.Response(
            200,
            json={"ok": True, "has_more": False, "messages": [{"text": "ドット歯磨きのラフ"}]},
        )
    raise AssertionError(path)


def test_import_channel_messages_walks_channels_and_filters_noise() -> None:
    from voice_ai_summary.slack_import import import_channel_messages

    with httpx.Client(transport=httpx.MockTransport(_channel_handler)) as client_http:
        lines = import_channel_messages(client_http, "xoxp-test", days=30)

    assert lines == [
        "#general",
        "安田さんに見積もりを送りました",
        "般務連絡です",
        "#design",
        "ドット歯磨きのラフ",
    ]


def test_import_channel_messages_respects_channel_filter_and_caps() -> None:
    from voice_ai_summary.slack_import import import_channel_messages

    with httpx.Client(transport=httpx.MockTransport(_channel_handler)) as client_http:
        only_design = import_channel_messages(client_http, "xoxp-test", channels=["#design"])
        capped = import_channel_messages(client_http, "xoxp-test", per_channel=1)

    assert only_design == ["#design", "ドット歯磨きのラフ"]
    assert capped == ["#general", "安田さんに見積もりを送りました", "#design", "ドット歯磨きのラフ"]


def test_extract_glossary_uses_channel_prompt_for_other_authors(monkeypatch) -> None:
    captured: list[str] = []

    class _Parsed:
        parsed_output = Glossary(terms=[Term(term="ドット歯磨き")])
        usage = None

    class _Messages:
        def parse(self, **kwargs):
            captured.append(kwargs["system"])
            return _Parsed()

    class _Client:
        messages = _Messages()

    own = extract_glossary(_Client(), "m", ["自分の発言"], Glossary())
    chan = extract_glossary(
        _Client(), "m", ["#general", "他人の発言"], Glossary(), own_messages=False
    )

    assert "ユーザー本人のSlackメッセージ" in captured[0]
    assert "参加しているSlackチャンネル" in captured[1]
    assert "style_notes は出力しないでください" in captured[1]
    assert [t.term for t in own.terms] == [t.term for t in chan.terms] == ["ドット歯磨き"]


def test_vocab_import_merges_json_file(vas, tmp_path) -> None:
    from typer.testing import CliRunner

    from voice_ai_summary.cli import app
    from voice_ai_summary.glossary import load_glossary, save_glossary

    cfg, _conn = vas
    save_glossary(cfg, Glossary(terms=[Term(term="安田さん", note="取引先")]))
    src = tmp_path / "slack_glossary.json"
    src.write_text(
        json.dumps(
            {
                "terms": [
                    {"term": "安田さん", "aliases": ["安田"], "note": ""},
                    {"term": "ドット歯磨き", "aliases": [], "note": "新製品の企画名"},
                    {"term": ""},
                ],
                "style_notes": ["社名はカタカナで書く"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["vocab", "import", str(src)])

    assert result.exit_code == 0, result.output
    g = load_glossary(cfg)
    by_term = {t.term: t for t in g.terms}
    assert set(by_term) == {"安田さん", "ドット歯磨き"}
    assert by_term["安田さん"].note == "取引先" and "安田" in by_term["安田さん"].aliases
    assert g.style_notes == ["社名はカタカナで書く"]
    assert "imported 2 term(s)" in result.output
