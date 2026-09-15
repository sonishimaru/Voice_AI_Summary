"""Tests for the personal glossary: storage, merge/format logic, Slack import, and
extraction. No network calls - `import_from_slack` uses `httpx.MockTransport` and
`extract_glossary` is tested with a monkeypatched `_call_extract`."""

from __future__ import annotations

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

    def fake_call_extract(client, model, text, existing_block) -> Glossary:
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
