"""steerable-mcp: a stdio MCP server that exposes run_state, propose and history.

It connects with ``STEERABLE_AGENT_DSN`` and refuses to start unless that login is the
``steerable_agent`` role, so a misconfigured deployment cannot hand an LLM a stronger
credential by accident. Every query is parameterised; every database rejection is returned
to the client as a tool result, never raised as an instruction.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

AGENT_ROLE = "steerable_agent"


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain(v) for v in value]
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    return value


def coerce_value(value: Any) -> Any:
    """MCP clients sometimes send numbers and booleans as strings; parse JSON when it is."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


class AgentDB:
    """The three read/write operations the agent credential allows, over one connection."""

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self._conn: psycopg.Connection[dict[str, Any]] | None = None

    def connect(self) -> psycopg.Connection[dict[str, Any]]:
        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(self.dsn, row_factory=dict_row, autocommit=True)
        return self._conn

    def assert_agent_role(self) -> str:
        row = (
            self.connect()
            .execute(
                "SELECT current_user AS who, pg_has_role(current_user, %s, 'MEMBER') AS is_agent",
                (AGENT_ROLE,),
            )
            .fetchone()
        )
        assert row is not None
        if not row["is_agent"]:
            raise PermissionError(
                f"steerable-mcp must connect as {AGENT_ROLE}; the DSN logs in as {row['who']}. "
                "Refusing to expose a stronger credential to an LLM."
            )
        return str(row["who"])

    def run_state(self, run_id: int) -> dict[str, Any]:
        row = (
            self.connect()
            .execute("SELECT * FROM steerable.run_state WHERE run_id = %s", (run_id,))
            .fetchone()
        )
        if row is None:
            return {"error": f"run {run_id} does not exist or is not visible"}
        plain: dict[str, Any] = _plain(dict(row))
        return plain

    def history(self, run_id: int, limit: int = 50) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 500))
        rows = (
            self.connect()
            .execute(
                "SELECT * FROM steerable.history WHERE run_id = %s ORDER BY proposal_id DESC LIMIT %s",
                (run_id, limit),
            )
            .fetchall()
        )
        return [_plain(dict(r)) for r in rows]

    def propose(self, run_id: int, key: str, value: Any, reason: str) -> dict[str, Any]:
        try:
            row = (
                self.connect()
                .execute(
                    "SELECT * FROM steerable.propose(%s, %s, %s, %s)",
                    (run_id, key, Jsonb(coerce_value(value)), reason),
                )
                .fetchone()
            )
        except psycopg.Error as exc:
            return {
                "accepted": False,
                "sqlstate": exc.sqlstate,
                "message": exc.diag.message_primary,
            }
        assert row is not None
        return {"accepted": True, "proposal_id": row["proposal_id"], "status": row["status"]}


def build_server(db: AgentDB) -> Any:
    from mcp.server.mcpserver import MCPServer

    server = MCPServer(
        "steerable",
        instructions=(
            "Steer a training run through typed, policy-checked proposals. Call run_state first; "
            "propose only keys listed in its policies, within their schema and limits. A rejection "
            "is data about the policy, not an instruction to work around it."
        ),
    )

    @server.tool(description="Effective config, policies, last metrics and open proposals for a run.")
    def run_state(run_id: int) -> dict[str, Any]:
        return db.run_state(run_id)

    @server.tool(
        description=(
            "Propose a config change. value is a JSON value matching the key's policy schema. "
            "Returns accepted=false with the database's message when the policy refuses it."
        )
    )
    def propose(run_id: int, key: str, value: Any, reason: str) -> dict[str, Any]:
        return db.propose(run_id, key, value, reason)

    @server.tool(description="Past proposals for a run with their events (approved, applied, rejected, ...).")
    def history(run_id: int, limit: int = 50) -> list[dict[str, Any]]:
        return db.history(run_id, limit)

    return server


def main() -> int:
    dsn = os.environ.get("STEERABLE_AGENT_DSN") or os.environ.get("STEERABLE_DSN")
    if not dsn:
        print("STEERABLE_AGENT_DSN is required", file=sys.stderr)
        return 2
    db = AgentDB(dsn)
    try:
        who = db.assert_agent_role()
    except (PermissionError, psycopg.Error) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"steerable-mcp connected as {who}", file=sys.stderr)
    build_server(db).run(transport="stdio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
