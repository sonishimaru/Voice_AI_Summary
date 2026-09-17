"""Worker shutdown behaviour: a stop signal is not a transcription failure."""

from __future__ import annotations

import wave
from pathlib import Path

import pytest

from voice_ai_summary import retention, worker
from voice_ai_summary import vad as vad_module
from voice_ai_summary.config import Config
from voice_ai_summary.db import utcnow_iso
from voice_ai_summary.ingest import ingest_file
from voice_ai_summary.pipeline import process_recording
from voice_ai_summary.recorder_state import RecorderState
from voice_ai_summary.vad import SpeechRegion


def _recorder_state(**overrides: object) -> RecorderState:
    fields = {
        "state": "recording",
        "since": "2026-09-17T09:00:00Z",
        "resume_at": None,
        "reason": "user",
        "pid": 4321,
        "app_version": "0.2.0",
        "updated_at": "2026-09-17T09:00:00Z",
    }
    fields.update(overrides)
    return RecorderState(**fields)


def _write_wav(path: Path, seconds: float = 2.0, sample_rate: int = 16000) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * int(seconds * sample_rate))


class _RaisingBackend:
    """Fails the way the worker sees failures: from inside `transcribe`."""

    name = "raising"

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    def transcribe(
        self, samples: object, language: str | None = None, source: str | None = None
    ) -> list:
        raise self._exc


def _one_recording(vas: tuple[Config, object], monkeypatch: pytest.MonkeyPatch) -> tuple:
    cfg, conn = vas
    src = cfg.paths.inbox / "mac_system_dev1_20260915T000000Z.wav"
    _write_wav(src)
    rec_id = ingest_file(conn, cfg, src)
    monkeypatch.setattr(
        vad_module, "detect_speech", lambda samples, cfg: [SpeechRegion(0, 1000, 0.9)]
    )
    return cfg, conn, rec_id


def test_stop_does_not_become_a_recording_error(
    vas: tuple[Config, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`launchctl kickstart -k` SIGTERMs the worker on every restart, so a stop landing
    mid-transcription is the common path. It must not blame the recording."""
    cfg, conn, rec_id = _one_recording(vas, monkeypatch)

    # Simulate the live claim `process_pending` would hold on this row while it is being
    # decoded - exactly the state a SIGTERM lands in mid-transcription.
    conn.execute("UPDATE recordings SET claimed_at = ? WHERE id = ?", (utcnow_iso(), rec_id))
    conn.commit()

    with pytest.raises(worker._Stop):
        process_recording(conn, cfg, rec_id, _RaisingBackend(worker._Stop()))

    row = conn.execute(
        "SELECT error, processed_at, claimed_at FROM recordings WHERE id = ?", (rec_id,)
    ).fetchone()
    assert row["error"] is None
    assert row["processed_at"] is None
    # Regression: `_Stop` is a `BaseException` (deliberately, so it isn't swallowed as a
    # transcription failure), but that also means the old `except Exception` handler -
    # the only thing that cleared `claimed_at` - never ran for it either. Without the
    # fix this claim survives the interrupt and strands the row for `CLAIM_TIMEOUT_S`
    # (45 minutes) before any worker will touch it again.
    assert row["claimed_at"] is None


def test_a_real_failure_is_still_recorded(
    vas: tuple[Config, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg, conn, rec_id = _one_recording(vas, monkeypatch)

    conn.execute("UPDATE recordings SET claimed_at = ? WHERE id = ?", (utcnow_iso(), rec_id))
    conn.commit()

    with pytest.raises(ValueError):
        process_recording(conn, cfg, rec_id, _RaisingBackend(ValueError("decode failed")))

    row = conn.execute(
        "SELECT error, claimed_at FROM recordings WHERE id = ?", (rec_id,)
    ).fetchone()
    assert "decode failed" in row["error"]
    # Ordinary failures must keep clearing the claim exactly as before.
    assert row["claimed_at"] is None


class TestBacklogWatch:
    """`_BacklogWatch`: fire once per backlog episode, re-arm only after recovery."""

    def test_fires_once_then_rearms_after_recovery(self) -> None:
        # Consumed in order: two calls while first arming (call 1: arm, call 2: below
        # grace), one call that crosses the grace period (fires), one call while already
        # fired (must stay quiet), then two more once a second episode starts.
        clock_values = iter([0, 40, 70, 80, 200, 300])
        fired: list[tuple[int, float]] = []

        watch = worker._BacklogWatch(
            count_threshold=5,
            grace_s=60,
            clock=lambda: next(clock_values),
            notify=lambda pending, elapsed: fired.append((pending, elapsed)),
        )

        watch.observe(6)  # t=0: backlog starts
        watch.observe(6)  # t=40: below grace period, no fire yet
        watch.observe(6)  # t=70: past grace period -> fires
        assert fired == [(6, 70)]

        watch.observe(6)  # t=80: still backlogged, already fired -> stays quiet
        assert fired == [(6, 70)]

        watch.observe(2)  # recovers: drops below threshold, no clock call, re-arms
        watch.observe(6)  # t=200: new episode starts
        watch.observe(6)  # t=300: past grace period again -> fires a second time

        assert fired == [(6, 70), (6, 100)]

    def test_disabled_when_threshold_or_grace_is_zero(self) -> None:
        fired: list[tuple[int, float]] = []
        watch = worker._BacklogWatch(
            count_threshold=0,
            grace_s=60,
            clock=lambda: 999,
            notify=lambda pending, elapsed: fired.append((pending, elapsed)),
        )
        for _ in range(5):
            watch.observe(100)
        assert fired == []


class TestRecorderWatch:
    """`_RecorderWatch`: fire once when the recorder claims `recording` (and looks
    alive) but no inbox file has arrived for >= 3x the rotation interval; re-arm on
    the next file arrival."""

    def _watch(
        self,
        *,
        clock_values: list[float],
        state: RecorderState | None,
        alive: bool = True,
        fired: list[float] | None = None,
        rotation_s: float = 60.0,
    ) -> tuple[worker._RecorderWatch, list[float]]:
        fired = fired if fired is not None else []
        clock_iter = iter(clock_values)
        watch = worker._RecorderWatch(
            rotation_s=rotation_s,
            clock=lambda: next(clock_iter),
            notify=lambda elapsed: fired.append(elapsed),
            read_state=lambda: state,
            is_alive=lambda s: alive,
        )
        return watch, fired

    def test_fires_once_after_three_missed_rotations_while_recording_and_alive(self) -> None:
        # rotation_s=60 -> threshold is 180s. t=0 arms (first sighting), t=100 below
        # threshold, t=181 crosses it -> fires, t=181 (same instant/next poll) already
        # fired -> stays quiet.
        watch, fired = self._watch(
            clock_values=[0, 100, 181, 181],
            state=_recorder_state(),
        )
        watch.observe(False)  # t=0: first sighting, starts the clock
        watch.observe(False)  # t=100: below the 180s threshold
        assert fired == []
        watch.observe(False)  # t=181: past threshold -> fires
        assert fired == [181]
        watch.observe(False)  # already fired -> stays quiet
        assert fired == [181]

    def test_file_arrival_rearms(self) -> None:
        watch, fired = self._watch(
            clock_values=[0, 181, 181, 400],
            state=_recorder_state(),
        )
        watch.observe(False)  # t=0: arms
        watch.observe(False)  # t=181: 181s since arrival -> fires
        assert fired == [181]
        watch.observe(True)  # t=181: a file arrives -> resets `_last_arrival` to 181
        watch.observe(False)  # t=400: 219s since the reset (>=180s) -> fires again,
        # proving the watch re-armed rather than staying permanently silent.
        assert fired == [181, 219]

    def test_silent_when_paused(self) -> None:
        watch, fired = self._watch(
            clock_values=[0, 1, 2, 3],
            state=_recorder_state(state="paused"),
        )
        for _ in range(4):
            watch.observe(False)
        assert fired == []

    def test_silent_when_stopped(self) -> None:
        watch, fired = self._watch(
            clock_values=[0, 1, 2, 3],
            state=_recorder_state(state="stopped"),
        )
        for _ in range(4):
            watch.observe(False)
        assert fired == []

    def test_silent_when_no_state_file(self) -> None:
        watch, fired = self._watch(clock_values=[0, 1, 2, 3], state=None)
        for _ in range(4):
            watch.observe(False)
        assert fired == []

    def test_silent_when_recording_but_process_dead(self) -> None:
        watch, fired = self._watch(
            clock_values=[0, 181, 181],
            state=_recorder_state(),
            alive=False,
        )
        for _ in range(3):
            watch.observe(False)
        assert fired == []

    def test_notify_raising_does_not_propagate(self) -> None:
        def failing_notify(elapsed: float) -> None:
            raise RuntimeError("osascript boom")

        clock_iter = iter([0, 181])
        watch = worker._RecorderWatch(
            rotation_s=60.0,
            clock=lambda: next(clock_iter),
            notify=failing_notify,
            read_state=lambda: _recorder_state(),
            is_alive=lambda s: True,
        )
        watch.observe(False)  # arms
        watch.observe(False)  # would fire, notify raises - must not propagate


def test_worker_survives_broken_recorder_state(
    vas: tuple[Config, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A recorder-state read that raises must not stop the worker loop."""
    cfg, _conn = vas

    monkeypatch.setattr(worker.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(worker, "process_pending", lambda conn, cfg, backend: 0)

    def raising_read_state(root: object) -> None:
        raise RuntimeError("torn state file")

    monkeypatch.setattr(worker, "read_state", raising_read_state)

    calls = {"n": 0}

    def bounded_ingest(conn: object, cfg: object) -> list:
        calls["n"] += 1
        if calls["n"] > 2:
            raise worker._Stop
        return []

    monkeypatch.setattr(worker, "ingest_inbox", bounded_ingest)

    clock_values = iter([0.0, 10.0])
    worker.run_worker(cfg, clock=lambda: next(clock_values))
    assert calls["n"] == 3


def test_worker_survives_notification_failure(
    vas: tuple[Config, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A backlog notification that raises must not stop the worker loop."""
    cfg, _conn = vas
    cfg.schedule.backlog_alert_count = 1
    cfg.schedule.backlog_alert_minutes = 1  # 60s grace

    monkeypatch.setattr(worker.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(worker, "process_pending", lambda conn, cfg, backend: 0)
    monkeypatch.setattr(worker, "_pending_count", lambda conn: 5)  # always past threshold

    def failing_notify(pending: int, elapsed: float) -> None:
        raise RuntimeError("osascript boom")

    monkeypatch.setattr(worker, "_notify_backlog", failing_notify)

    calls = {"n": 0}

    def bounded_ingest(conn: object, cfg: object) -> list:
        # Bound the loop deterministically (no real time.sleep, no real osascript):
        # let the alert fire once, then stop the worker via the normal shutdown path.
        calls["n"] += 1
        if calls["n"] > 2:
            raise worker._Stop
        return []

    monkeypatch.setattr(worker, "ingest_inbox", bounded_ingest)

    # An unbounded incrementing fake clock: `run_worker` now also reads the clock to
    # schedule `retention.prune` (see TestRetentionSchedule below), so the loop consumes
    # more ticks per iteration than just `_BacklogWatch.observe`'s one - a fixed-length
    # `iter([...])` would run out and raise `StopIteration` instead of exercising the
    # shutdown path this test is actually about.
    clock_state = {"t": 0.0}

    def _clock() -> float:
        clock_state["t"] += 50.0
        return clock_state["t"]

    # Must return normally (the _Stop is caught inside run_worker) despite the
    # notification raising on the second poll.
    worker.run_worker(cfg, clock=_clock)


class TestRetentionSchedule:
    """`run_worker` also runs `retention.prune` on its own schedule - once an hour,
    driven by the same injectable `clock` as `_BacklogWatch`, not on every poll."""

    def test_prune_runs_once_an_hour_not_every_poll(
        self, vas: tuple[Config, object], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg, _conn = vas
        # Keep `_BacklogWatch.observe` from consuming clock ticks of its own: it only
        # calls the clock once `pending >= count_threshold`, and there's nothing
        # pending here (a fresh `vas` fixture, threshold defaults to 10).
        monkeypatch.setattr(worker.time, "sleep", lambda seconds: None)
        monkeypatch.setattr(worker, "process_pending", lambda conn, cfg, backend: 0)

        calls = {"n": 0}

        def bounded_ingest(conn: object, cfg: object) -> list:
            calls["n"] += 1
            if calls["n"] > 3:
                raise worker._Stop
            return []

        monkeypatch.setattr(worker, "ingest_inbox", bounded_ingest)

        prune_calls: list[tuple] = []

        def fake_prune(conn: object, cfg: object, *, dry_run: bool, log_dir, skip_logs) -> None:
            prune_calls.append((dry_run, skip_logs))

        monkeypatch.setattr(retention, "prune", fake_prune)

        # 3 successful polls: t=0 (first ever poll -> prunes), t=10 (10s later, well
        # under an hour -> skipped), t=4000 (past the hour mark from t=0 -> prunes again).
        clock_values = iter([0.0, 10.0, 4000.0])
        worker.run_worker(cfg, clock=lambda: next(clock_values))

        assert len(prune_calls) == 2
        assert all(dry_run is False for dry_run, _ in prune_calls)
        assert all(skip_logs == retention.WORKER_LOG_BASENAMES for _, skip_logs in prune_calls)

    def test_a_raising_prune_does_not_stop_the_worker_loop(
        self, vas: tuple[Config, object], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg, _conn = vas
        monkeypatch.setattr(worker.time, "sleep", lambda seconds: None)
        monkeypatch.setattr(worker, "process_pending", lambda conn, cfg, backend: 0)

        calls = {"n": 0}

        def bounded_ingest(conn: object, cfg: object) -> list:
            calls["n"] += 1
            if calls["n"] > 2:
                raise worker._Stop
            return []

        monkeypatch.setattr(worker, "ingest_inbox", bounded_ingest)

        def raising_prune(conn: object, cfg: object, **kwargs: object) -> None:
            raise RuntimeError("prune boom")

        monkeypatch.setattr(retention, "prune", raising_prune)

        clock_values = iter([0.0, 10.0])
        # Must return normally: run_worker logs the exception and keeps polling instead
        # of letting a retention bug take down the whole worker.
        worker.run_worker(cfg, clock=lambda: next(clock_values))
        assert calls["n"] == 3
