"""Tests for inbox ingestion."""

from __future__ import annotations

import json
import os
import time
import wave
from pathlib import Path

import pytest

from voice_ai_summary import ingest as ingest_mod
from voice_ai_summary.config import Config
from voice_ai_summary.ingest import ingest_file, ingest_inbox, parse_inbox_name


def _write_wav(path: Path, seconds: float = 0.5, sample_rate: int = 16000) -> None:
    n_samples = int(seconds * sample_rate)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_samples)


def _age(path: Path, seconds: float = 10.0) -> None:
    """Backdate a file's mtime past the "still being written" guard."""
    old = time.time() - seconds
    os.utime(path, (old, old))


def test_parse_inbox_name_matches_recorder_pattern() -> None:
    parsed = parse_inbox_name(Path("mac_mic_device1_20260915T010203Z.m4a"))
    assert parsed == {
        "source": "mac_mic",
        "device_id": "device1",
        "started_at_utc": "2026-09-15T01:02:03Z",
    }


def test_parse_inbox_name_no_match() -> None:
    assert parse_inbox_name(Path("not_a_recording.wav")) is None


def test_ingest_file_moves_into_store_and_is_idempotent(vas: tuple[Config, object]) -> None:
    cfg, conn = vas
    src = cfg.paths.inbox / "mac_mic_dev1_20260915T010203Z.wav"
    _write_wav(src)

    rec_id = ingest_file(conn, cfg, src)
    assert rec_id is not None
    assert not src.exists()

    row = conn.execute("SELECT * FROM recordings WHERE id = ?", (rec_id,)).fetchone()
    assert row["source"] == "mac_mic"
    assert row["device_id"] == "dev1"
    assert row["started_at_utc"] == "2026-09-15T01:02:03Z"
    stored = cfg.paths.store / row["storage_path"]
    assert stored.is_file()

    # Ingest the same bytes again under a new name: idempotent, no duplicate row.
    src2 = cfg.paths.inbox / "mac_mic_dev1_20260915T010203Z_dup.wav"
    _write_wav(src2)
    rec_id2 = ingest_file(conn, cfg, src2)
    assert rec_id2 == rec_id
    assert not src2.exists()
    count = conn.execute("SELECT COUNT(*) AS n FROM recordings").fetchone()["n"]
    assert count == 1


def test_ingest_file_reads_sidecar_json(vas: tuple[Config, object]) -> None:
    cfg, conn = vas
    src = cfg.paths.inbox / "weird_name.wav"
    _write_wav(src)
    sidecar = src.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "source": "mac_system",
                "device_id": "mac-1",
                "started_at_utc": "2026-01-02T03:04:05Z",
                "tz_offset": "+09:00",
            }
        ),
        encoding="utf-8",
    )

    rec_id = ingest_file(conn, cfg, src)
    row = conn.execute("SELECT * FROM recordings WHERE id = ?", (rec_id,)).fetchone()
    assert row["source"] == "mac_system"
    assert row["device_id"] == "mac-1"
    assert row["started_at_utc"] == "2026-01-02T03:04:05Z"
    assert row["tz_offset"] == "+09:00"
    assert not sidecar.exists()


def test_ingest_file_fallback_metadata(vas: tuple[Config, object]) -> None:
    cfg, conn = vas
    src = cfg.paths.inbox / "recording.wav"
    _write_wav(src)

    rec_id = ingest_file(conn, cfg, src)
    row = conn.execute("SELECT * FROM recordings WHERE id = ?", (rec_id,)).fetchone()
    assert row["source"] == "file"
    assert row["started_at_utc"]  # derived from file mtime


def test_ingest_inbox_processes_all_files(vas: tuple[Config, object]) -> None:
    cfg, conn = vas
    a = cfg.paths.inbox / "mac_mic_dev1_20260915T010203Z.wav"
    b = cfg.paths.inbox / "mac_system_dev1_20260915T020304Z.wav"
    _write_wav(a, seconds=0.5)
    _write_wav(b, seconds=0.6)  # different content so it hashes differently
    _age(a)
    _age(b)

    ids = ingest_inbox(conn, cfg)
    assert len(ids) == 2
    count = conn.execute("SELECT COUNT(*) AS n FROM recordings").fetchone()["n"]
    assert count == 2


def test_ingest_inbox_skips_recently_modified_files(vas: tuple[Config, object]) -> None:
    cfg, conn = vas
    _write_wav(cfg.paths.inbox / "mac_mic_dev1_20260915T010203Z.wav")

    ids = ingest_inbox(conn, cfg)
    assert ids == []


def test_ingest_inbox_skips_in_progress_and_hidden_files(vas: tuple[Config, object]) -> None:
    cfg, conn = vas
    part = cfg.paths.inbox / "mac_mic_dev1_20260915T010203Z.m4a.part"
    hidden = cfg.paths.inbox / ".DS_Store"
    _write_wav(part)
    hidden.write_bytes(b"x")
    old = time.time() - 60
    os.utime(part, (old, old))
    os.utime(hidden, (old, old))

    assert ingest_inbox(conn, cfg) == []
    assert part.exists()


def test_ingest_file_stores_at_0600(vas: tuple[Config, object]) -> None:
    import stat

    cfg, conn = vas
    src = cfg.paths.inbox / "mac_mic_dev1_20260915T010203Z.wav"
    _write_wav(src)
    # Simulate a file the recorder wrote before any private-umask/mode discipline
    # existed - a mode `ingest_file` must actively fix, not just happen to inherit.
    src.chmod(0o644)

    rec_id = ingest_file(conn, cfg, src)
    row = conn.execute("SELECT storage_path FROM recordings WHERE id = ?", (rec_id,)).fetchone()
    stored = cfg.paths.store / row["storage_path"]

    assert stat.S_IMODE(stored.stat().st_mode) == 0o600


def test_ingest_inbox_skips_a_file_that_vanishes_mid_pass(
    vas: tuple[Config, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The recorder can delete/rotate a file between `ingest_inbox`'s listing and the
    per-file ingest (e.g. its own cleanup) - the worker must not crash-loop over that
    race, and the other, unaffected file must still be ingested."""
    import voice_ai_summary.ingest as ingest_module

    cfg, conn = vas
    vanishing = cfg.paths.inbox / "mac_mic_dev1_20260915T010203Z.wav"
    survives = cfg.paths.inbox / "mac_system_dev1_20260915T020304Z.wav"
    _write_wav(vanishing, seconds=0.5)
    _write_wav(survives, seconds=0.6)
    _age(vanishing)
    _age(survives)

    real_ingest_file = ingest_module.ingest_file

    def flaky_ingest_file(conn, cfg, path, **kwargs):
        if path.name == vanishing.name:
            raise FileNotFoundError(path)
        return real_ingest_file(conn, cfg, path, **kwargs)

    monkeypatch.setattr(ingest_module, "ingest_file", flaky_ingest_file)

    ids = ingest_inbox(conn, cfg)

    assert len(ids) == 1
    count = conn.execute("SELECT COUNT(*) AS n FROM recordings").fetchone()["n"]
    assert count == 1


class TestRecoverStaleParts:
    """`.part` files the recorder abandoned (crash, force-quit, a finalize that
    never ran) hold real audio that ingestion skips by design, so nothing ever
    mentions it again. Recovery turns that silent loss into a delay."""

    def _write(self, cfg, name: str, *, age_s: float) -> Path:
        path = cfg.paths.inbox / name
        path.write_bytes(b"audio")
        stamp = time.time() - age_s
        os.utime(path, (stamp, stamp))
        return path

    def test_renames_a_part_nobody_is_writing_to(self, vas) -> None:
        cfg, _ = vas
        stale = self._write(cfg, "mac_mic_dev_20260917T100000Z.m4a.part", age_s=60 * 60)

        recovered = ingest_mod.recover_stale_parts(cfg)

        assert recovered == [cfg.paths.inbox / "mac_mic_dev_20260917T100000Z.m4a"]
        assert not stale.exists()
        assert recovered[0].read_bytes() == b"audio"

    def test_leaves_a_file_the_recorder_still_has_open(self, vas) -> None:
        cfg, _ = vas
        fresh = self._write(cfg, "mac_mic_dev_20260917T100000Z.m4a.part", age_s=30)

        assert ingest_mod.recover_stale_parts(cfg) == []
        assert fresh.exists()

    def test_cutoff_follows_the_configured_rotation(self, vas) -> None:
        cfg, _ = vas
        cfg.recorder.rotation_minutes = 1
        # Older than 3 x 1 min, but well inside the 3 x 15 min default.
        self._write(cfg, "mac_mic_dev_20260917T100000Z.m4a.part", age_s=5 * 60)

        assert len(ingest_mod.recover_stale_parts(cfg)) == 1

    def test_does_not_overwrite_an_existing_final_file(self, vas) -> None:
        cfg, _ = vas
        stale = self._write(cfg, "mac_mic_dev_20260917T100000Z.m4a.part", age_s=60 * 60)
        final = cfg.paths.inbox / "mac_mic_dev_20260917T100000Z.m4a"
        final.write_bytes(b"the good copy")

        assert ingest_mod.recover_stale_parts(cfg) == []
        assert stale.exists()
        assert final.read_bytes() == b"the good copy"

    def test_ingest_inbox_picks_up_what_it_recovered(self, vas) -> None:
        cfg, conn = vas
        self._write(cfg, "mac_mic_dev_20260917T100000Z.m4a.part", age_s=60 * 60)

        ids = ingest_inbox(conn, cfg)

        assert len(ids) == 1
        row = conn.execute("SELECT source, started_at_utc FROM recordings").fetchone()
        assert row["source"] == "mac_mic"
        assert row["started_at_utc"] == "2026-09-17T10:00:00Z"
