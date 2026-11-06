"""The SQL contract: grants, immutability, every propose() rejection path, supersede, the fold."""

from __future__ import annotations

from typing import Any

import psycopg
import pytest
from psycopg.types.json import Jsonb

from conftest import LAUNCH_CONFIG, Roles, connect, create_run
from steerable.types import Policy

Conn = psycopg.Connection[dict[str, Any]]


def propose(conn: Conn, run_id: int, key: str, value: Any, reason: str = "test") -> dict[str, Any]:
    row = conn.execute(
        "SELECT * FROM steerable.propose(%s, %s, %s, %s)", (run_id, key, Jsonb(value), reason)
    ).fetchone()
    assert row is not None
    return row


def status(conn: Conn, proposal_id: int) -> str:
    row = conn.execute(
        "SELECT status FROM steerable.proposal_status WHERE proposal_id = %s", (proposal_id,)
    ).fetchone()
    assert row is not None
    return str(row["status"])


def config(conn: Conn, run_id: int) -> dict[str, Any]:
    rows = conn.execute(
        "SELECT key, value FROM steerable.current_config WHERE run_id = %s", (run_id,)
    ).fetchall()
    return {r["key"]: r["value"] for r in rows}


def rejects(conn: Conn, sqlstate: str, sql: str, params: tuple[Any, ...] = ()) -> str:
    with pytest.raises(psycopg.Error) as info:
        conn.execute(sql, params)
    assert info.value.sqlstate == sqlstate, f"{info.value.sqlstate}: {info.value}"
    return str(info.value.diag.message_primary)


# ----- grants -----


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM steerable.runs",
        "SELECT * FROM steerable.policies",
        "SELECT * FROM steerable.proposals",
        "SELECT * FROM steerable.events",
        "SELECT * FROM steerable.metrics",
        "SELECT * FROM steerable.pending",
        "SELECT * FROM steerable.current_config",
        "UPDATE steerable.proposals SET value = '0' WHERE id = 1",
        "DELETE FROM steerable.events WHERE id = 1",
        "INSERT INTO steerable.policies (run_id, key, schema) VALUES (1, 'x', '{\"type\":\"boolean\"}')",
        "INSERT INTO steerable.metrics (run_id, step, name, value) VALUES (1, 1, 'loss', 1)",
        "UPDATE steerable.runs SET agent_enabled = true",
        "SELECT steerable.ack(1, 1, 'applied', NULL)",
        "SELECT steerable.decide(1, 'approved', NULL)",
        "SELECT steerable.launch(1, '{}', '[]')",
        "SELECT steerable.finish(1, 'finished')",
    ],
)
def test_agent_has_no_table_or_admin_access(agent: Conn, sql: str) -> None:
    rejects(agent, "42501", sql)


def test_agent_can_read_state_and_history_and_propose(agent: Conn, run: int) -> None:
    state = agent.execute("SELECT * FROM steerable.run_state WHERE run_id = %s", (run,)).fetchone()
    assert state is not None and state["config"]["lr"] == 1e-3
    assert set(state.keys()) == {
        "run_id",
        "name",
        "status",
        "agent_enabled",
        "step",
        "config",
        "policies",
        "metrics",
        "open_proposals",
    }
    assert "note" not in state["policies"]["resolution"]
    row = propose(agent, run, "lr", 5e-4, "halve lr")
    assert row["status"] == "approved"
    hist = agent.execute("SELECT * FROM steerable.history WHERE run_id = %s", (run,)).fetchall()
    assert [h["proposal_id"] for h in hist] == [row["proposal_id"]]
    assert hist[0]["actor_kind"] == "agent"
    assert "actor" not in hist[0]


def test_agent_cannot_create_objects_in_schema(agent: Conn) -> None:
    rejects(agent, "42501", "CREATE TABLE steerable.evil (x int)")
    rejects(agent, "42501", "CREATE FUNCTION steerable.evil() RETURNS int LANGUAGE sql AS 'SELECT 1'")


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM steerable.propose(1, 'lr', '0.1', 'x')",
        "SELECT steerable.decide(1, 'approved', NULL)",
        "UPDATE steerable.policies SET mode = 'auto'",
        "UPDATE steerable.runs SET agent_enabled = false",
        "INSERT INTO steerable.policies (run_id, key, schema) VALUES (1, 'x', '{\"type\":\"boolean\"}')",
        "INSERT INTO steerable.proposals (run_id, key, value, actor, actor_kind, reason) "
        "VALUES (1, 'lr', '1', 'me', 'system', 'x')",
        "INSERT INTO steerable.events (proposal_id, kind, actor) VALUES (1, 'applied', 'me')",
        "DELETE FROM steerable.metrics",
        "UPDATE steerable.metrics SET value = 0",
    ],
)
def test_trainer_cannot_propose_decide_or_change_policy(trainer: Conn, sql: str) -> None:
    rejects(trainer, "42501", sql)


def test_trainer_reads_applies_and_logs(trainer: Conn, agent: Conn, run: int) -> None:
    pid = propose(agent, run, "lr", 5e-4)["proposal_id"]
    pend = trainer.execute("SELECT * FROM steerable.pending WHERE run_id = %s", (run,)).fetchall()
    assert [p["proposal_id"] for p in pend] == [pid] and pend[0]["boundary"] == "step"
    trainer.execute(
        "INSERT INTO steerable.metrics (run_id, step, name, value) VALUES (%s, 10, 'loss', 0.5)", (run,)
    )
    trainer.execute("SELECT steerable.ack(%s, 10, 'applied', NULL)", (pid,))
    assert status(trainer, pid) == "applied"
    assert config(trainer, run)["lr"] == 5e-4


# ----- immutability -----


def test_proposals_and_events_are_append_only_even_for_admin(admin: Conn, agent: Conn, run: int) -> None:
    pid = propose(agent, run, "lr", 5e-4)["proposal_id"]
    rejects(admin, "SE000", "UPDATE steerable.proposals SET value = '1' WHERE id = %s", (pid,))
    rejects(admin, "SE000", "DELETE FROM steerable.proposals WHERE id = %s", (pid,))
    rejects(admin, "SE000", "UPDATE steerable.events SET kind = 'rejected' WHERE proposal_id = %s", (pid,))
    rejects(admin, "SE000", "DELETE FROM steerable.events WHERE proposal_id = %s", (pid,))
    rejects(admin, "SE000", "TRUNCATE steerable.events")
    rejects(admin, "SE000", "TRUNCATE steerable.proposals CASCADE")


# ----- propose() rejection paths, in validation order -----


def test_rejects_bad_arguments(agent: Conn, run: int) -> None:
    rejects(agent, "SE009", "SELECT * FROM steerable.propose(%s, 'lr', '0.1', '')", (run,))
    rejects(agent, "SE009", "SELECT * FROM steerable.propose(%s, 'lr', NULL, 'x')", (run,))
    rejects(agent, "SE009", "SELECT * FROM steerable.propose(%s, 'lr', '0.1', %s)", (run, "r" * 1001))
    rejects(agent, "SE009", "SELECT * FROM steerable.propose(%s, %s, '0.1', 'x')", (run, "k" * 65))


def test_rejects_missing_or_not_running_run(agent: Conn, admin: Conn, trainer: Conn) -> None:
    msg = rejects(agent, "SE001", "SELECT * FROM steerable.propose(999999999, 'lr', '0.1', 'x')")
    assert "does not exist" in msg
    created = create_run(admin, trainer, launch=False)
    rejects(agent, "SE001", "SELECT * FROM steerable.propose(%s, 'lr', '0.1', 'x')", (created,))
    done = create_run(admin, trainer)
    trainer.execute("SELECT steerable.finish(%s, 'finished')", (done,))
    msg = rejects(agent, "SE001", "SELECT * FROM steerable.propose(%s, 'lr', '0.0005', 'x')", (done,))
    assert "finished" in msg


def test_rejects_frozen_run(agent: Conn, admin: Conn, run: int) -> None:
    admin.execute("UPDATE steerable.runs SET agent_enabled = false WHERE id = %s", (run,))
    msg = rejects(agent, "SE002", "SELECT * FROM steerable.propose(%s, 'lr', '0.0005', 'x')", (run,))
    assert "frozen" in msg
    admin.execute("UPDATE steerable.runs SET agent_enabled = true WHERE id = %s", (run,))
    assert propose(agent, run, "lr", 5e-4)["status"] == "approved"


def test_rejects_unknown_key(agent: Conn, run: int) -> None:
    msg = rejects(agent, "SE003", "SELECT * FROM steerable.propose(%s, 'model', '\"bigger\"', 'x')", (run,))
    assert "no policy" in msg
    rejects(agent, "SE003", "SELECT * FROM steerable.propose(%s, 'drop table', '1', 'x')", (run,))


def test_rejects_frozen_key_with_its_reason(agent: Conn, run: int) -> None:
    msg = rejects(agent, "SE004", "SELECT * FROM steerable.propose(%s, 'resolution', '1280', 'x')", (run,))
    assert "frozen" in msg and "sizes the head" in msg


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("lr", "0.5"),  # above max
        ("lr", "0"),  # below min
        ("lr", '"0.0005"'),  # string, not number
        ("lr", "true"),
        ("lr", "[0.0005]"),
        ("lr", '{"$eval": "1"}'),
        ("grad_accum_steps", "2.5"),  # not an integer
        ("grad_accum_steps", "0"),
        ("use_amp", "1"),
        ("use_amp", '"true"'),
        ("lr_scheduler", '"linear"'),
        ("lr_scheduler", "1"),
    ],
)
def test_rejects_values_outside_schema(agent: Conn, run: int, key: str, value: str) -> None:
    msg = rejects(
        agent, "SE005", f"SELECT * FROM steerable.propose(%s, %s, '{value}'::jsonb, 'x')", (run, key)
    )
    assert "does not match" in msg


def test_rejects_relative_change_beyond_policy(agent: Conn, run: int) -> None:
    msg = rejects(agent, "SE006", "SELECT * FROM steerable.propose(%s, 'lr', '0.01', 'x')", (run,))
    assert "10x" in msg and "max_rel_change is 3x" in msg
    rejects(agent, "SE006", "SELECT * FROM steerable.propose(%s, 'lr', '0.0001', 'x')", (run,))
    assert propose(agent, run, "lr", 3e-3)["status"] == "approved"  # exactly 3x is allowed


def test_relative_change_measured_against_effective_value(agent: Conn, trainer: Conn, run: int) -> None:
    pid = propose(agent, run, "lr", 3e-3)["proposal_id"]
    rejects(
        agent, "SE006", "SELECT * FROM steerable.propose(%s, 'lr', '0.009', 'x')", (run,)
    )  # still vs 1e-3
    trainer.execute("SELECT steerable.ack(%s, 1, 'applied', NULL)", (pid,))
    assert propose(agent, run, "lr", 9e-3)["status"] == "approved"  # now vs 3e-3


def test_rejects_sign_flip_or_zero_when_bounded(agent: Conn, admin: Conn, trainer: Conn) -> None:
    pols = [Policy("w", {"type": "number", "min": -10, "max": 10}, mode="auto", max_rel_change=2)]
    run = create_run(admin, trainer, config={"w": 1.0}, policies=pols)
    rejects(agent, "SE006", "SELECT * FROM steerable.propose(%s, 'w', '-1', 'x')", (run,))
    rejects(agent, "SE006", "SELECT * FROM steerable.propose(%s, 'w', '0', 'x')", (run,))
    run0 = create_run(admin, trainer, config={"w": 0.0}, policies=pols)
    assert propose(agent, run0, "w", 5.0)["status"] == "approved"  # no ratio from zero: schema bounds apply


def test_rejects_too_soon_after_last_applied(agent: Conn, trainer: Conn, run: int) -> None:
    pid = propose(agent, run, "grad_accum_steps", 2)["proposal_id"]
    trainer.execute("SELECT steerable.ack(%s, 50, 'applied', NULL)", (pid,))
    trainer.execute(
        "INSERT INTO steerable.metrics (run_id, step, name, value) VALUES (%s, 120, 'loss', 1)", (run,)
    )
    msg = rejects(agent, "SE007", "SELECT * FROM steerable.propose(%s, 'grad_accum_steps', '4', 'x')", (run,))
    assert "step 150" in msg
    trainer.execute(
        "INSERT INTO steerable.metrics (run_id, step, name, value) VALUES (%s, 150, 'loss', 1)", (run,)
    )
    assert propose(agent, run, "grad_accum_steps", 4)["status"] == "approved"


def test_rejects_sixth_proposal_in_an_hour_for_a_key(agent: Conn, run: int) -> None:
    for _ in range(5):
        propose(agent, run, "lr", 5e-4)
    msg = rejects(agent, "SE008", "SELECT * FROM steerable.propose(%s, 'lr', '0.0005', 'x')", (run,))
    assert "max_per_hour is 5" in msg


def test_rejects_run_level_rate_limit(agent: Conn, admin: Conn, trainer: Conn) -> None:
    run = create_run(admin, trainer, max_proposals_per_hour=2)
    propose(agent, run, "lr", 5e-4)
    propose(agent, run, "use_amp", True)
    msg = rejects(
        agent, "SE008", "SELECT * FROM steerable.propose(%s, 'lr_scheduler', '\"step\"', 'x')", (run,)
    )
    assert "run limit is 2" in msg


# ----- modes, decide, ack -----


def test_review_mode_waits_for_admin(agent: Conn, admin: Conn, trainer: Conn, run: int) -> None:
    row = propose(agent, run, "weight_decay", 1e-3)
    pid = row["proposal_id"]
    assert row["status"] == "pending"
    rejects(trainer, "SE010", "SELECT steerable.ack(%s, 1, 'applied', NULL)", (pid,))
    admin.execute("SELECT steerable.decide(%s, 'approved', 'fine')", (pid,))
    assert status(admin, pid) == "approved"
    rejects(admin, "SE010", "SELECT steerable.decide(%s, 'rejected', NULL)", (pid,))
    trainer.execute("SELECT steerable.ack(%s, 7, 'applied', NULL)", (pid,))
    assert status(admin, pid) == "applied"
    rejects(trainer, "SE010", "SELECT steerable.ack(%s, 8, 'applied', NULL)", (pid,))


def test_admin_rejection_is_terminal(agent: Conn, admin: Conn, trainer: Conn, run: int) -> None:
    pid = propose(agent, run, "weight_decay", 1e-3)["proposal_id"]
    admin.execute("SELECT steerable.decide(%s, 'rejected', 'no')", (pid,))
    assert status(admin, pid) == "rejected"
    rejects(trainer, "SE010", "SELECT steerable.ack(%s, 1, 'applied', NULL)", (pid,))
    assert config(admin, run)["weight_decay"] == 1e-4


def test_trainer_rejection_keeps_effective_value(agent: Conn, trainer: Conn, run: int) -> None:
    pid = propose(agent, run, "lr", 5e-4)["proposal_id"]
    trainer.execute("SELECT steerable.ack(%s, 3, 'trainer_rejected', 'ValueError: no optimizer')", (pid,))
    assert status(trainer, pid) == "trainer_rejected"
    assert config(trainer, run)["lr"] == 1e-3
    rejects(trainer, "SE009", "SELECT steerable.ack(%s, 3, 'approved', NULL)", (pid,))


def test_decide_status_validated(admin: Conn, agent: Conn, run: int) -> None:
    pid = propose(agent, run, "weight_decay", 1e-3)["proposal_id"]
    rejects(admin, "SE009", "SELECT steerable.decide(%s, 'applied', NULL)", (pid,))
    rejects(admin, "SE010", "SELECT steerable.decide(999999999, 'approved', NULL)")


# ----- supersede -----


def test_newer_proposal_supersedes_open_older_one(agent: Conn, trainer: Conn, run: int) -> None:
    first = propose(agent, run, "lr", 5e-4)["proposal_id"]
    second = propose(agent, run, "lr", 6e-4)["proposal_id"]
    assert status(trainer, first) == "superseded"
    assert status(trainer, second) == "approved"
    pend = trainer.execute("SELECT proposal_id FROM steerable.pending WHERE run_id = %s", (run,)).fetchall()
    assert [p["proposal_id"] for p in pend] == [second]
    third = propose(agent, run, "weight_decay", 2e-4)["proposal_id"]  # different key, pending in review
    fourth = propose(agent, run, "weight_decay", 3e-4)["proposal_id"]
    assert status(trainer, third) == "superseded" and status(trainer, fourth) == "pending"


def test_applied_proposal_is_not_superseded(agent: Conn, trainer: Conn, run: int) -> None:
    first = propose(agent, run, "lr", 5e-4)["proposal_id"]
    trainer.execute("SELECT steerable.ack(%s, 1, 'applied', NULL)", (first,))
    propose(agent, run, "lr", 6e-4)
    assert status(trainer, first) == "applied"


# ----- fold -----


def test_effective_config_folds_applied_events_over_launch(agent: Conn, trainer: Conn, run: int) -> None:
    assert config(trainer, run) == LAUNCH_CONFIG
    a = propose(agent, run, "lr", 5e-4)["proposal_id"]
    trainer.execute("SELECT steerable.ack(%s, 10, 'applied', NULL)", (a,))
    b = propose(agent, run, "lr", 2.5e-4)["proposal_id"]
    trainer.execute("SELECT steerable.ack(%s, 20, 'applied', NULL)", (b,))
    c = propose(agent, run, "use_amp", True)["proposal_id"]
    trainer.execute("SELECT steerable.ack(%s, 20, 'applied', NULL)", (c,))
    d = propose(agent, run, "lr_scheduler", "step")["proposal_id"]
    trainer.execute("SELECT steerable.ack(%s, 21, 'trainer_rejected', 'no')", (d,))
    expected = dict(LAUNCH_CONFIG, lr=2.5e-4, use_amp=True)
    assert config(trainer, run) == expected
    state = agent.execute("SELECT config, step FROM steerable.run_state WHERE run_id = %s", (run,)).fetchone()
    assert state is not None and state["config"] == expected and state["step"] == 21
    row = trainer.execute(
        "SELECT proposal_id, applied_step FROM steerable.current_config WHERE run_id = %s AND key = 'lr'",
        (run,),
    ).fetchone()
    assert row is not None and row["proposal_id"] == b and row["applied_step"] == 20
    assert config(trainer, run)["model"] == "toy"  # launch-only keys pass through


def test_previous_is_recorded_at_proposal_time(agent: Conn, trainer: Conn, run: int) -> None:
    pid = propose(agent, run, "lr", 5e-4)["proposal_id"]
    row = trainer.execute("SELECT previous FROM steerable.proposals WHERE id = %s", (pid,)).fetchone()
    assert row is not None and row["previous"] == 1e-3


# ----- launch -----


def test_launch_is_once_and_keeps_admin_seeded_policies(admin: Conn, trainer: Conn, agent: Conn) -> None:
    run = create_run(admin, trainer, launch=False)
    admin.execute(
        "INSERT INTO steerable.policies (run_id, key, schema, mode, max_rel_change) VALUES (%s, 'lr', %s, 'review', 1.5)",
        (run, Jsonb({"type": "number", "min": 0, "max": 1})),
    )
    trainer.execute(
        "SELECT steerable.launch(%s, %s, %s)",
        (
            run,
            Jsonb({"lr": 1e-3}),
            Jsonb([Policy("lr", {"type": "number"}, mode="auto", max_rel_change=100).to_json()]),
        ),
    )
    pol = trainer.execute(
        "SELECT mode, max_rel_change FROM steerable.policies WHERE run_id = %s", (run,)
    ).fetchone()
    assert pol is not None and pol["mode"] == "review" and float(pol["max_rel_change"]) == 1.5
    rejects(trainer, "SE001", "SELECT steerable.launch(%s, '{}', '[]')", (run,))
    rejects(trainer, "SE009", "SELECT steerable.finish(%s, 'bogus')", (run,))
    assert propose(agent, run, "lr", 1.2e-3)["status"] == "pending"


def test_launch_rejects_malformed_policy(admin: Conn, trainer: Conn) -> None:
    run = create_run(admin, trainer, launch=False)
    with pytest.raises(psycopg.Error):
        trainer.execute(
            "SELECT steerable.launch(%s, '{}', %s)",
            (run, Jsonb([{"key": "lr", "schema": {"type": "string"}}])),
        )
    with pytest.raises(psycopg.Error):
        trainer.execute(
            "SELECT steerable.launch(%s, '{}', %s)",
            (run, Jsonb([{"key": "DROP TABLE", "schema": {"type": "boolean"}}])),
        )


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "string"},
        {"type": "number", "min": "0"},
        {"type": "number", "min": 1, "max": 0},
        {"type": "enum", "values": []},
        {"type": "enum", "values": [{"a": 1}]},
        {"type": "enum"},
        [],
        "number",
    ],
)
def test_policy_schema_check(admin: Conn, run: int, schema: Any) -> None:
    rejects(
        admin,
        "23514",
        "INSERT INTO steerable.policies (run_id, key, schema) VALUES (%s, 'k', %s)",
        (run, Jsonb(schema)),
    )


# ----- run_state shape -----


def test_run_state_metrics_window_and_open_proposals(agent: Conn, trainer: Conn, run: int) -> None:
    with trainer.cursor() as cur:
        cur.executemany(
            "INSERT INTO steerable.metrics (run_id, step, name, value) VALUES (%s, %s, %s, %s)",
            [(run, s, "loss", 1.0 / (s + 1)) for s in range(40)] + [(run, 39, "map", 0.3)],
        )
    propose(agent, run, "weight_decay", 2e-4, "try more regularisation")
    state = agent.execute("SELECT * FROM steerable.run_state WHERE run_id = %s", (run,)).fetchone()
    assert state is not None
    assert len(state["metrics"]["loss"]) == 25 and state["metrics"]["loss"][-1][0] == 39
    assert state["metrics"]["map"] == [[39, 0.3]]
    assert state["step"] == 39
    assert [p["status"] for p in state["open_proposals"]] == ["pending"]
    assert state["open_proposals"][0]["reason"] == "try more regularisation"


def test_each_role_sees_only_its_own_surface(roles: Roles) -> None:
    def privileges(dsn: str) -> set[tuple[str, str]]:
        with connect(dsn) as conn:
            rows = conn.execute(
                """
                SELECT table_name, privilege_type FROM information_schema.role_table_grants
                WHERE table_schema = 'steerable' AND grantee = current_user
                """
            ).fetchall()
            funcs = conn.execute(
                """
                SELECT p.proname FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace
                WHERE n.nspname = 'steerable' AND has_function_privilege(current_user, p.oid, 'EXECUTE')
                      AND p.proname NOT IN ('schema_is_valid', 'value_matches', 'caller_kind',
                                            'refuse_mutation', 'touch_run')
                """
            ).fetchall()
        return {(r["table_name"], r["privilege_type"]) for r in rows} | {
            (f["proname"], "EXECUTE") for f in funcs
        }

    assert privileges(roles.agent) == {("run_state", "SELECT"), ("history", "SELECT"), ("propose", "EXECUTE")}
    trainer = privileges(roles.trainer)
    assert {p for p in trainer if p[1] != "SELECT"} == {
        ("metrics", "INSERT"),
        ("ack", "EXECUTE"),
        ("launch", "EXECUTE"),
        ("finish", "EXECUTE"),
    }
