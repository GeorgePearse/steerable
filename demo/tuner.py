"""Scripted tuner. Holds only the steerable_agent DSN: it can read run_state/history and call propose().

Halves the learning rate once the run passes step 200 and doubles the GIoU loss weight after
step 400. Every rejection from the database is printed verbatim, as data.
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


def propose(conn: psycopg.Connection[dict[str, Any]], run: int, key: str, value: Any, reason: str) -> None:
    try:
        row = conn.execute(
            "SELECT * FROM steerable.propose(%s, %s, %s, %s)", (run, key, Jsonb(value), reason)
        ).fetchone()
        assert row is not None
        print(f"proposed {key}={value}: proposal {row['proposal_id']} {row['status']}", flush=True)
    except psycopg.Error as exc:
        print(f"rejected {key}={value} [{exc.sqlstate}]: {exc.diag.message_primary}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True, help="steerable_agent DSN")
    ap.add_argument("--run", type=int, required=True)
    ap.add_argument("--poll-s", type=float, default=0.25)
    ap.add_argument("--timeout-s", type=float, default=600.0)
    args = ap.parse_args()

    plan = [
        (200, "lr", lambda cfg: cfg["lr"] / 2, "loss plateaued over the last 100 steps; halving lr"),
        (400, "loss_giou", lambda cfg: cfg["loss_giou"] * 2, "val IoU lags the L1 term; weight GIoU more"),
    ]
    done: set[str] = set()
    deadline = time.monotonic() + args.timeout_s
    with psycopg.connect(args.dsn, row_factory=dict_row, autocommit=True) as conn:
        while time.monotonic() < deadline:
            state = conn.execute(
                "SELECT * FROM steerable.run_state WHERE run_id = %s", (args.run,)
            ).fetchone()
            if state is None:
                print(f"run {args.run} not visible yet", flush=True)
            elif state["status"] in ("finished", "aborted"):
                print(f"run {args.run} is {state['status']}; tuner exiting", flush=True)
                break
            elif state["status"] == "running" and state["step"] is not None:
                for at_step, key, new_value, reason in plan:
                    if key not in done and state["step"] >= at_step:
                        done.add(key)
                        propose(conn, args.run, key, new_value(state["config"]), reason)
            time.sleep(args.poll_s)
        else:
            print(
                f"tuner timed out after {args.timeout_s}s with {sorted(set(k for _, k, _, _ in plan) - done)} undone",
                flush=True,
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
