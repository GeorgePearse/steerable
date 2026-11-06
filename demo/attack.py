"""`make attack`: everything a hostile or confused agent might try with the steerable_agent credential.

Each attempt must fail with the database's own error. The admin and trainer DSNs are used only
to set the stage (create and launch a run); every attack runs as steerable_agent.
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from steerable.types import Policy

POLICIES = [
    Policy("lr", {"type": "number", "min": 1e-6, "max": 1e-1}, mode="auto", max_rel_change=3, max_per_hour=5),
    Policy(
        "loss_giou", {"type": "number", "min": 0, "max": 20}, mode="auto", max_rel_change=3, max_per_hour=5
    ),
    Policy(
        "image_size",
        {"type": "integer", "min": 32, "max": 256},
        boundary="frozen",
        note="fixes the input tensor shape",
    ),
]


def attempt(
    conn: psycopg.Connection[dict[str, Any]], label: str, sql: str, params: tuple[Any, ...] = ()
) -> bool:
    try:
        rows = conn.execute(sql, params).fetchall()
    except psycopg.Error as exc:
        print(f"{label}\n    -> {exc.sqlstate} {exc.diag.message_primary}", flush=True)
        return True
    print(f"{label}\n    -> SUCCEEDED (unexpected): {rows[:3]}", flush=True)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--admin-dsn", required=True)
    ap.add_argument("--trainer-dsn", required=True)
    ap.add_argument("--agent-dsn", required=True)
    args = ap.parse_args()

    name = f"attack-{time.strftime('%Y%m%d-%H%M%S')}"
    with psycopg.connect(args.admin_dsn, autocommit=True) as admin:
        row = admin.execute("INSERT INTO steerable.runs (name) VALUES (%s) RETURNING id", (name,)).fetchone()
        assert row is not None
        run = int(row[0])
    with psycopg.connect(args.trainer_dsn, autocommit=True) as trainer:
        trainer.execute(
            "SELECT steerable.launch(%s, %s, %s)",
            (
                run,
                Jsonb({"lr": 1e-3, "loss_giou": 2.0, "image_size": 64}),
                Jsonb([p.to_json() for p in POLICIES]),
            ),
        )
    print(f"stage: run {run} is running with lr=0.001 (max_rel_change 3, max_per_hour 5)\n", flush=True)

    refused = 0
    total = 0
    with psycopg.connect(args.agent_dsn, row_factory=dict_row, autocommit=True) as agent:
        who = agent.execute("SELECT current_user").fetchone()
        assert who is not None
        print(f"connected as {who['current_user']}\n", flush=True)
        attacks: list[tuple[str, str, tuple[Any, ...]]] = [
            (
                "UPDATE steerable.proposals SET value = '0.1'",
                "UPDATE steerable.proposals SET value = '0.1'",
                (),
            ),
            ("DELETE FROM steerable.events", "DELETE FROM steerable.events", ()),
            (
                "INSERT INTO steerable.policies (...)",
                "INSERT INTO steerable.policies (run_id, key, schema, mode) VALUES (%s, 'lr', '{\"type\":\"number\"}', 'auto')",
                (run,),
            ),
            ("SELECT * FROM steerable.runs", "SELECT * FROM steerable.runs", ()),
            ("SELECT * FROM steerable.proposals", "SELECT * FROM steerable.proposals", ()),
            (
                "UPDATE steerable.runs SET agent_enabled = true",
                "UPDATE steerable.runs SET agent_enabled = true",
                (),
            ),
            ("SELECT steerable.decide(1, 'approved')", "SELECT steerable.decide(1, 'approved', NULL)", ()),
            ("SELECT steerable.ack(1, 1, 'applied')", "SELECT steerable.ack(1, 1, 'applied', NULL)", ()),
            ("CREATE TABLE steerable.notes (t text)", "CREATE TABLE steerable.notes (t text)", ()),
            (
                "propose unknown key 'optimizer'",
                "SELECT * FROM steerable.propose(%s, 'optimizer', '\"sgd\"', 'try sgd')",
                (run,),
            ),
            (
                "propose lr = 0.5 (schema max 0.1)",
                "SELECT * FROM steerable.propose(%s, 'lr', '0.5', 'go faster')",
                (run,),
            ),
            (
                'propose lr = "0.0005" (string, not number)',
                "SELECT * FROM steerable.propose(%s, 'lr', '\"0.0005\"', 'halve')",
                (run,),
            ),
            (
                "propose lr = 0.01 (10x jump, max_rel_change 3)",
                "SELECT * FROM steerable.propose(%s, 'lr', '0.01', '10x')",
                (run,),
            ),
            (
                "propose frozen key image_size = 128",
                "SELECT * FROM steerable.propose(%s, 'image_size', '128', 'bigger')",
                (run,),
            ),
            ("propose with empty reason", "SELECT * FROM steerable.propose(%s, 'lr', '0.0005', '')", (run,)),
            (
                "propose on a run that does not exist",
                "SELECT * FROM steerable.propose(999999, 'lr', '0.0005', 'x')",
                (),
            ),
        ]
        for label, sql, params in attacks:
            total += 1
            refused += attempt(agent, label, sql, params)

        print("\nfive valid lr proposals, then a sixth inside the hour (max_per_hour 5):", flush=True)
        for i in range(5):
            row = agent.execute(
                "SELECT * FROM steerable.propose(%s, 'lr', %s, %s)",
                (run, Jsonb(0.0009 - i * 0.0001), f"nudge {i + 1}"),
            ).fetchone()
            assert row is not None
            print(f"    proposal {row['proposal_id']} {row['status']}", flush=True)
        total += 1
        refused += attempt(
            agent,
            "propose lr sixth time",
            "SELECT * FROM steerable.propose(%s, 'lr', '0.0004', 'sixth')",
            (run,),
        )

        print(
            "\nadmin freezes the run (UPDATE runs SET agent_enabled = false); the agent tries again:",
            flush=True,
        )
        with psycopg.connect(args.admin_dsn, autocommit=True) as admin:
            admin.execute("UPDATE steerable.runs SET agent_enabled = false WHERE id = %s", (run,))
        total += 1
        refused += attempt(
            agent,
            "propose loss_giou = 3 on the frozen run",
            "SELECT * FROM steerable.propose(%s, 'loss_giou', '3', 'more giou')",
            (run,),
        )

    print(f"\n{refused}/{total} attempts refused by the database", flush=True)
    return 0 if refused == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
