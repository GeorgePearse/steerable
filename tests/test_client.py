"""Steerable client against a real database: fetch, dry-run, apply-all-or-reject, ack, hold, fold."""

from __future__ import annotations

from typing import Any

import psycopg
import pytest
from psycopg.types.json import Jsonb

from conftest import DEFAULT_POLICIES, LAUNCH_CONFIG, Roles, create_run
from steerable import Steerable

Conn = psycopg.Connection[dict[str, Any]]


class Recorder:
    def __init__(self, fail_dry_run: bool = False) -> None:
        self.calls: list[tuple[Any, bool]] = []
        self.fail_dry_run = fail_dry_run

    def __call__(self, value: Any, *, dry_run: bool) -> None:
        self.calls.append((value, dry_run))
        if self.fail_dry_run and dry_run:
            raise ValueError(f"cannot take {value}")


def propose(agent: Conn, run: int, key: str, value: Any) -> int:
    row = agent.execute(
        "SELECT * FROM steerable.propose(%s, %s, %s, 'test')", (run, key, Jsonb(value))
    ).fetchone()
    assert row is not None
    return int(row["proposal_id"])


def status(conn: Conn, pid: int) -> tuple[str, int | None, str | None]:
    row = conn.execute(
        "SELECT status, status_step, status_note FROM steerable.proposal_status WHERE proposal_id = %s",
        (pid,),
    ).fetchone()
    assert row is not None
    return row["status"], row["status_step"], row["status_note"]


@pytest.fixture
def steer(roles: Roles, run: int) -> Any:
    lr, amp, bs = Recorder(), Recorder(), Recorder()
    s = Steerable(roles.trainer, run, {"lr": lr, "use_amp": amp, "batch_size": bs}, poll_interval_s=0.1)
    s.recorders = {"lr": lr, "use_amp": amp, "batch_size": bs}  # type: ignore[attr-defined]
    yield s
    s.close()


def test_value_reflects_launch_config(steer: Steerable) -> None:
    assert steer.value("lr") == 1e-3
    assert steer.value("model") == "toy"
    assert steer.value("missing", 7) == 7


def test_sync_applies_dry_run_first_and_acks(steer: Steerable, agent: Conn, trainer: Conn, run: int) -> None:
    pid = propose(agent, run, "lr", 5e-4)
    assert steer.wait_for_proposals(5)
    applied = steer.sync(step=42)
    assert [p.id for p in applied] == [pid]
    rec = steer.recorders["lr"]  # type: ignore[attr-defined]
    assert rec.calls == [(5e-4, True), (5e-4, False)]
    assert status(trainer, pid) == ("applied", 42, None)
    assert steer.value("lr") == 5e-4
    assert steer.sync(step=43) == []


def test_dry_run_failure_rejects_that_proposal_and_applies_the_rest(
    roles: Roles, run: int, agent: Conn, trainer: Conn
) -> None:
    bad, good = Recorder(fail_dry_run=True), Recorder()
    with Steerable(roles.trainer, run, {"lr": bad, "use_amp": good}, poll_interval_s=0.1) as s:
        p_lr = propose(agent, run, "lr", 5e-4)
        p_amp = propose(agent, run, "use_amp", True)
        assert s.wait_for_proposals(5)
        import time

        time.sleep(0.3)
        applied = s.sync(step=5)
    assert [p.id for p in applied] == [p_amp]
    assert bad.calls == [(5e-4, True)]
    assert good.calls == [(True, True), (True, False)]
    assert status(trainer, p_lr) == ("trainer_rejected", 5, "ValueError: cannot take 0.0005")
    assert status(trainer, p_amp) == ("applied", 5, None)


def test_unknown_key_is_trainer_rejected(roles: Roles, run: int, agent: Conn, trainer: Conn) -> None:
    with Steerable(roles.trainer, run, {}, poll_interval_s=0.1) as s:
        pid = propose(agent, run, "lr", 5e-4)
        assert s.wait_for_proposals(5)
        assert s.sync(step=1) == []
    st, _, note = status(trainer, pid)
    assert st == "trainer_rejected" and note == "no applier registered for key lr"


def test_validation_boundary_is_held_until_validation(
    steer: Steerable, agent: Conn, trainer: Conn, run: int
) -> None:
    pid = propose(agent, run, "batch_size", 8)
    assert steer.wait_for_proposals(5)
    assert steer.sync(step=10) == []
    assert status(trainer, pid)[0] == "approved"
    assert steer.value("batch_size") == 16
    applied = steer.sync(step=20, at_validation=True)
    assert [p.id for p in applied] == [pid]
    assert status(trainer, pid) == ("applied", 20, None)
    assert steer.value("batch_size") == 8


def test_only_latest_per_key_in_a_batch_is_applied(
    steer: Steerable, agent: Conn, trainer: Conn, run: int
) -> None:
    first = propose(agent, run, "lr", 5e-4)
    assert steer.wait_for_proposals(5)
    second = propose(agent, run, "lr", 4e-4)
    import time

    time.sleep(0.3)
    applied = steer.sync(step=3)
    assert [p.id for p in applied] == [second]
    assert status(trainer, first)[0] == "superseded"
    assert status(trainer, second) == ("applied", 3, None)
    assert steer.recorders["lr"].calls == [(4e-4, True), (4e-4, False)]  # type: ignore[attr-defined]


def test_metrics_are_flushed_in_the_background(steer: Steerable, trainer: Conn, run: int) -> None:
    for step in range(5):
        steer.log_metric(step, "loss", 1.0 / (step + 1))
    import time

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        n = trainer.execute(
            "SELECT count(*) AS n FROM steerable.metrics WHERE run_id = %s", (run,)
        ).fetchone()
        assert n is not None
        if n["n"] == 5:
            break
        time.sleep(0.05)
    assert n["n"] == 5


def test_launch_and_finish(roles: Roles, admin: Conn, trainer: Conn) -> None:
    run = create_run(admin, trainer, launch=False)
    with Steerable(roles.trainer, run, {}, poll_interval_s=0.1) as s:
        assert s.value("lr") is None
        s.launch(LAUNCH_CONFIG, DEFAULT_POLICIES)
        assert s.value("lr") == 1e-3
        row = trainer.execute("SELECT status FROM steerable.runs WHERE id = %s", (run,)).fetchone()
        assert row is not None and row["status"] == "running"
        s.finish("finished")
    row = trainer.execute("SELECT status FROM steerable.runs WHERE id = %s", (run,)).fetchone()
    assert row is not None and row["status"] == "finished"


def test_new_instance_resumes_effective_config(roles: Roles, run: int, agent: Conn) -> None:
    rec = Recorder()
    with Steerable(roles.trainer, run, {"lr": rec}, poll_interval_s=0.1) as s:
        propose(agent, run, "lr", 5e-4)
        assert s.wait_for_proposals(5)
        s.sync(step=1)
    with Steerable(roles.trainer, run, {"lr": rec}, poll_interval_s=0.1) as resumed:
        assert resumed.value("lr") == 5e-4
