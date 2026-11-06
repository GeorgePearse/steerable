"""Command line: ``steerable db init | run create | propose | approve | reject | freeze | unfreeze | history | state | tune``.

The DSN decides what you can do. ``propose`` works with the agent or admin DSN; ``approve``,
``reject``, ``freeze``, ``unfreeze`` and ``run create`` need the admin DSN; ``db init`` needs a
superuser DSN. Nothing here bypasses the database's own checks.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


def _dsn(args: argparse.Namespace) -> str:
    dsn = args.dsn or os.environ.get("STEERABLE_DSN")
    if not dsn:
        sys.exit("a DSN is required: --dsn or STEERABLE_DSN")
    return str(dsn)


def _json_default(o: Any) -> Any:
    if isinstance(o, datetime | date):
        return o.isoformat()
    if isinstance(o, Decimal):
        return float(o)
    return str(o)


def _dump(rows: Any) -> None:
    print(json.dumps(rows, default=_json_default, indent=2))


def _parse_value(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _connect(dsn: str) -> psycopg.Connection[dict[str, Any]]:
    return psycopg.connect(dsn, row_factory=dict_row, autocommit=True)


def cmd_db_init(args: argparse.Namespace) -> int:
    from steerable import db

    superuser = args.superuser_dsn or os.environ.get("STEERABLE_SUPERUSER_DSN")
    if not superuser:
        sys.exit("--superuser-dsn or STEERABLE_SUPERUSER_DSN is required")
    passwords = {}
    for role in ("admin", "trainer", "agent"):
        value = getattr(args, f"{role}_password") or os.environ.get(f"STEERABLE_{role.upper()}_PASSWORD")
        if not value:
            sys.exit(f"--{role}-password or STEERABLE_{role.upper()}_PASSWORD is required")
        passwords[f"{role}_password"] = value
    created = db.init(superuser, reset=args.reset, **passwords)
    print("schema steerable created" if created else "schema steerable already present; roles updated")
    return 0


def cmd_run_create(args: argparse.Namespace) -> int:
    with _connect(_dsn(args)) as conn:
        row = conn.execute(
            "INSERT INTO steerable.runs (name, max_proposals_per_hour) VALUES (%s, %s) RETURNING id",
            (args.name, args.max_proposals_per_hour),
        ).fetchone()
    assert row is not None
    print(row["id"])
    return 0


def cmd_propose(args: argparse.Namespace) -> int:
    with _connect(_dsn(args)) as conn:
        try:
            row = conn.execute(
                "SELECT * FROM steerable.propose(%s, %s, %s, %s)",
                (args.run, args.key, Jsonb(_parse_value(args.value)), args.reason),
            ).fetchone()
        except psycopg.Error as exc:
            print(f"rejected [{exc.sqlstate}]: {exc.diag.message_primary}")
            return 1
    assert row is not None
    print(f"proposal {row['proposal_id']} {row['status']}")
    return 0


def _decide(args: argparse.Namespace, status: str) -> int:
    with _connect(_dsn(args)) as conn:
        try:
            conn.execute("SELECT steerable.decide(%s, %s, %s)", (args.proposal, status, args.note))
        except psycopg.Error as exc:
            print(f"refused [{exc.sqlstate}]: {exc.diag.message_primary}")
            return 1
    print(f"proposal {args.proposal} {status}")
    return 0


def cmd_approve(args: argparse.Namespace) -> int:
    return _decide(args, "approved")


def cmd_reject(args: argparse.Namespace) -> int:
    return _decide(args, "rejected")


def _set_agent(args: argparse.Namespace, enabled: bool) -> int:
    with _connect(_dsn(args)) as conn:
        cur = conn.execute("UPDATE steerable.runs SET agent_enabled = %s WHERE id = %s", (enabled, args.run))
        if cur.rowcount == 0:
            print(f"run {args.run} does not exist")
            return 1
    print(f"run {args.run} agent_enabled = {str(enabled).lower()}")
    return 0


def cmd_freeze(args: argparse.Namespace) -> int:
    return _set_agent(args, False)


def cmd_unfreeze(args: argparse.Namespace) -> int:
    return _set_agent(args, True)


def cmd_history(args: argparse.Namespace) -> int:
    with _connect(_dsn(args)) as conn:
        rows = conn.execute(
            "SELECT * FROM steerable.history WHERE run_id = %s ORDER BY proposal_id DESC LIMIT %s",
            (args.run, args.limit),
        ).fetchall()
    _dump(rows)
    return 0


def cmd_state(args: argparse.Namespace) -> int:
    with _connect(_dsn(args)) as conn:
        row = conn.execute("SELECT * FROM steerable.run_state WHERE run_id = %s", (args.run,)).fetchone()
    if row is None:
        print(f"run {args.run} does not exist")
        return 1
    _dump(row)
    return 0


def cmd_tune(args: argparse.Namespace) -> int:
    from steerable import tune

    return int(tune.main_from_args(args))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="steerable", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dsn", help="connection string (or STEERABLE_DSN)")
    sub = parser.add_subparsers(dest="command", required=True)

    db = sub.add_parser("db", help="database setup").add_subparsers(dest="db_command", required=True)
    init = db.add_parser("init", help="create roles, schema, functions and grants")
    init.add_argument("--superuser-dsn")
    init.add_argument("--admin-password")
    init.add_argument("--trainer-password")
    init.add_argument("--agent-password")
    init.add_argument("--reset", action="store_true", help="drop and recreate the schema (destroys history)")
    init.set_defaults(func=cmd_db_init)

    run = sub.add_parser("run", help="runs").add_subparsers(dest="run_command", required=True)
    create = run.add_parser("create", help="create a run in state 'created'; the trainer launches it")
    create.add_argument("name")
    create.add_argument("--max-proposals-per-hour", type=int, default=30)
    create.set_defaults(func=cmd_run_create)

    p = sub.add_parser("propose", help="propose a change through steerable.propose()")
    p.add_argument("--run", type=int, required=True)
    p.add_argument("key")
    p.add_argument("value", help="JSON value (a bare word is taken as a string)")
    p.add_argument("--reason", required=True)
    p.set_defaults(func=cmd_propose)

    for name, func in (("approve", cmd_approve), ("reject", cmd_reject)):
        d = sub.add_parser(name, help=f"{name} a pending proposal (admin)")
        d.add_argument("proposal", type=int)
        d.add_argument("--note")
        d.set_defaults(func=func)

    for name, func in (("freeze", cmd_freeze), ("unfreeze", cmd_unfreeze)):
        f = sub.add_parser(name, help=f"{name} agent proposals for a run (admin)")
        f.add_argument("--run", type=int, required=True)
        f.set_defaults(func=func)

    h = sub.add_parser("history", help="proposals and their events")
    h.add_argument("--run", type=int, required=True)
    h.add_argument("--limit", type=int, default=50)
    h.set_defaults(func=cmd_history)

    s = sub.add_parser("state", help="effective config, policies, recent metrics, open proposals")
    s.add_argument("--run", type=int, required=True)
    s.set_defaults(func=cmd_state)

    t = sub.add_parser("tune", help="reference autonomous tuner (steerable[tune])")
    t.add_argument("--run", type=int, required=True)
    t.add_argument(
        "--live", action="store_true", help="call the Anthropic API (costs money); default is a dry run"
    )
    t.add_argument("--rounds", type=int, default=1)
    t.add_argument("--interval-s", type=float, default=60.0)
    t.add_argument("--model", default=None)
    t.add_argument("--max-usd", type=float, default=2.0)
    t.set_defaults(func=cmd_tune)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
