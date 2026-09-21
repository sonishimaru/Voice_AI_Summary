"""Tests for the Claude-based correction pass. Network calls are stubbed by
monkeypatching `_call_correct` directly - no anthropic client ever touches the network."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

from voice_ai_summary import correct as correct_mod
from voice_ai_summary.config import Config
from voice_ai_summary.correct import CorrectionResult, Fix, correct_day
from voice_ai_summary.search import search

DAY = "2026-09-15"


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
    text: str,
    speaker: str = "me",
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
        (recording_id, t_start_ms, t_start_ms + 500, abs_start, text, speaker),
    )
    return cur.lastrowid


def _seed_utterances(conn: sqlite3.Connection, texts: list[str], day: str = DAY) -> list[int]:
    started = f"{day}T00:00:00Z"
    rec = _insert_recording(conn, source="mac_mic", started_at_utc=started, sha256="e" * 64)
    conn.commit()
    ids = [
        _insert_utterance(
            conn, recording_id=rec, rec_started_at_utc=started, t_start_ms=i * 1_000, text=text
        )
        for i, text in enumerate(texts)
    ]
    conn.commit()
    return ids


def test_correct_day_applies_fix_and_marks_others_corrected(vas, monkeypatch) -> None:
    cfg: Config
    cfg, conn = vas
    ids = _seed_utterances(conn, ["かたまです", "今日は会議です", "ありがとうございます"])

    def fake_call_correct(client, model, block, glossary_block):
        assert str(ids[0]) in block
        return CorrectionResult(fixes=[Fix(id=ids[0], text="西丸です")])

    monkeypatch.setattr(correct_mod, "_call_correct", fake_call_correct)

    changed = correct_day(conn, cfg, DAY, client=object())
    assert changed == 1

    fixed = conn.execute("SELECT * FROM utterances WHERE id = ?", (ids[0],)).fetchone()
    assert fixed["text"] == "西丸です"
    assert fixed["raw_text"] == "かたまです"
    assert fixed["corrected_at"] is not None
    assert fixed["correction_model"] == cfg.correct.model

    for other_id in ids[1:]:
        row = conn.execute("SELECT * FROM utterances WHERE id = ?", (other_id,)).fetchone()
        assert row["raw_text"] is None
        assert row["corrected_at"] is not None
        assert row["correction_model"] == cfg.correct.model


def test_correct_day_second_call_makes_zero_calls(vas, monkeypatch) -> None:
    cfg, conn = vas
    _seed_utterances(conn, ["テスト発話"])

    calls = {"n": 0}

    def fake_call_correct(client, model, block, glossary_block):
        calls["n"] += 1
        return CorrectionResult(fixes=[])

    monkeypatch.setattr(correct_mod, "_call_correct", fake_call_correct)

    assert correct_day(conn, cfg, DAY, client=object()) == 0
    assert calls["n"] == 1

    assert correct_day(conn, cfg, DAY, client=object()) == 0
    assert calls["n"] == 1


def test_correct_day_force_recalls(vas, monkeypatch) -> None:
    cfg, conn = vas
    _seed_utterances(conn, ["テスト発話"])

    calls = {"n": 0}

    def fake_call_correct(client, model, block, glossary_block):
        calls["n"] += 1
        return CorrectionResult(fixes=[])

    monkeypatch.setattr(correct_mod, "_call_correct", fake_call_correct)

    correct_day(conn, cfg, DAY, client=object())
    assert calls["n"] == 1

    correct_day(conn, cfg, DAY, client=object(), force=True)
    assert calls["n"] == 2


def test_correct_day_batches_split_by_batch_chars(vas, monkeypatch) -> None:
    cfg, conn = vas
    cfg.correct.batch_chars = 40
    ids = _seed_utterances(conn, [f"発話その{i}をテストします" for i in range(6)])

    seen_ids: list[int] = []
    call_count = {"n": 0}

    def fake_call_correct(client, model, block, glossary_block):
        call_count["n"] += 1
        for line in block.splitlines():
            seen_ids.append(int(line.split("\t")[0]))
        return CorrectionResult(fixes=[])

    monkeypatch.setattr(correct_mod, "_call_correct", fake_call_correct)

    correct_day(conn, cfg, DAY, client=object())

    assert call_count["n"] > 1
    assert seen_ids == ids


def test_correct_day_ignores_unknown_or_empty_fix(vas, monkeypatch) -> None:
    cfg, conn = vas
    ids = _seed_utterances(conn, ["発話A", "発話B"])

    def fake_call_correct(client, model, block, glossary_block):
        return CorrectionResult(
            fixes=[Fix(id=999999, text="存在しないID"), Fix(id=ids[1], text="")]
        )

    monkeypatch.setattr(correct_mod, "_call_correct", fake_call_correct)

    changed = correct_day(conn, cfg, DAY, client=object())
    assert changed == 0

    for utt_id, expected_text in zip(ids, ["発話A", "発話B"], strict=True):
        row = conn.execute("SELECT * FROM utterances WHERE id = ?", (utt_id,)).fetchone()
        assert row["text"] == expected_text
        assert row["raw_text"] is None
        assert row["corrected_at"] is not None


def test_correct_day_empty_day_returns_zero_without_client(vas, monkeypatch) -> None:
    cfg, conn = vas

    def boom(*args, **kwargs):
        raise AssertionError("anthropic.Anthropic() should not be constructed")

    monkeypatch.setattr(correct_mod.anthropic, "Anthropic", boom)

    assert correct_day(conn, cfg, "2026-09-16") == 0


def test_correct_day_search_finds_corrected_word(vas, monkeypatch) -> None:
    cfg, conn = vas
    ids = _seed_utterances(conn, ["かたまです"])

    def fake_call_correct(client, model, block, glossary_block):
        return CorrectionResult(fixes=[Fix(id=ids[0], text="西丸です")])

    monkeypatch.setattr(correct_mod, "_call_correct", fake_call_correct)
    correct_day(conn, cfg, DAY, client=object())

    results = search(conn, "西丸")
    assert any(r["text"] == "西丸です" for r in results)
    assert not search(conn, "かたま")


def test_correct_day_splits_batch_when_output_truncated(vas, monkeypatch) -> None:
    from voice_ai_summary.glossary import OutputTruncated

    cfg, conn = vas
    _seed_utterances(conn, ["いち", "に", "さん", "よん", "ご"])
    cfg.correct.batch_chars = 100_000
    seen: list[int] = []

    def fake_call_correct(client, model, block, glossary_block):
        n = len(block.split("\n"))
        seen.append(n)
        if n > 2:
            raise OutputTruncated("cut off")
        first_id = int(block.split("\t", 1)[0])
        return CorrectionResult(fixes=[Fix(id=first_id, text="修正済み")])

    monkeypatch.setattr(correct_mod, "_call_correct", fake_call_correct)
    changed = correct_day(conn, cfg, DAY, client=object())

    assert seen == [5, 2, 3, 1, 2]
    assert changed >= 1
    pending = conn.execute(
        "SELECT COUNT(*) AS n FROM utterances WHERE corrected_at IS NULL"
    ).fetchone()["n"]
    assert pending == 0


def test_force_recorrects_from_the_original_asr_text(vas, monkeypatch) -> None:
    """A second pass must see the original ASR text, not the first pass's output, so
    corrections cannot stack on top of each other."""
    cfg, conn = vas
    ids = _seed_utterances(conn, ["もうギョウ太郎死んでるよ"])
    seen: list[str] = []

    def fake_call_correct(client, model, block, glossary_block):
        seen.append(block)
        return CorrectionResult(fixes=[Fix(id=ids[0], text="もう行太郎死んでるよ")])

    monkeypatch.setattr(correct_mod, "_call_correct", fake_call_correct)
    correct_day(conn, cfg, DAY, client=object())
    correct_day(conn, cfg, DAY, client=object(), force=True)

    assert "ギョウ太郎" in seen[0] and "ギョウ太郎" in seen[1]
    row = conn.execute("SELECT text, raw_text FROM utterances WHERE id = ?", (ids[0],)).fetchone()
    assert row["text"] == "もう行太郎死んでるよ"
    assert row["raw_text"] == "もうギョウ太郎死んでるよ"


def test_correct_user_content_wraps_transcript_and_glossary() -> None:
    from voice_ai_summary.correct import _correct_user_content

    block = "1\t00:00\t[me]\tこんにちは"
    content = _correct_user_content(block, "## 用語集\n- 西丸")

    assert content.startswith(f"<transcript>\n{block}\n</transcript>")
    assert "<glossary>\n## 用語集\n- 西丸\n</glossary>" in content


def test_correct_user_content_omits_glossary_tag_when_empty() -> None:
    from voice_ai_summary.correct import _correct_user_content

    content = _correct_user_content("block text", "")

    assert content == "<transcript>\nblock text\n</transcript>"
    assert "<glossary>" not in content


def test_correct_system_carries_data_framing_sentence() -> None:
    from voice_ai_summary.correct import _CORRECT_SYSTEM

    assert "<transcript>" in _CORRECT_SYSTEM
    assert "<glossary>" in _CORRECT_SYSTEM
    assert "指示" in _CORRECT_SYSTEM


def test_force_undoes_a_fix_the_model_no_longer_makes(vas, monkeypatch) -> None:
    cfg, conn = vas
    ids = _seed_utterances(conn, ["歯で糸をやる"])

    monkeypatch.setattr(
        correct_mod,
        "_call_correct",
        lambda *a: CorrectionResult(fixes=[Fix(id=ids[0], text="歯でいとをやる")]),
    )
    correct_day(conn, cfg, DAY, client=object())

    monkeypatch.setattr(correct_mod, "_call_correct", lambda *a: CorrectionResult())
    changed = correct_day(conn, cfg, DAY, client=object(), force=True)

    assert changed == 0
    row = conn.execute("SELECT text, raw_text FROM utterances WHERE id = ?", (ids[0],)).fetchone()
    assert row["text"] == "歯で糸をやる"
    assert row["raw_text"] is None
