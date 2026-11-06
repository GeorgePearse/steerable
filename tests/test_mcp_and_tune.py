"""MCP server tools and the reference tuner, with the real database and a fake Anthropic client."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest

from conftest import Roles
from steerable import tune
from steerable.mcp_server import AgentDB, build_server, coerce_value

# ----- MCP server -----


def test_mcp_refuses_non_agent_credentials(roles: Roles) -> None:
    with pytest.raises(PermissionError, match="must connect as steerable_agent"):
        AgentDB(roles.admin).assert_agent_role()
    with pytest.raises(PermissionError):
        AgentDB(roles.trainer).assert_agent_role()
    assert AgentDB(roles.agent).assert_agent_role() == "steerable_agent"


def test_mcp_tools_round_trip(roles: Roles, run: int) -> None:
    db = AgentDB(roles.agent)
    server = build_server(db)
    tools = asyncio.run(server.list_tools())
    assert sorted(t.name for t in tools) == ["history", "propose", "run_state"]

    state = asyncio.run(server.call_tool("run_state", {"run_id": run}))
    content = state.structured_content or {}
    assert content["config"]["lr"] == 1e-3 and content["status"] == "running"

    ok = asyncio.run(
        server.call_tool("propose", {"run_id": run, "key": "lr", "value": "0.0005", "reason": "halve"})
    )
    assert ok.structured_content == {
        "accepted": True,
        "proposal_id": ok.structured_content["proposal_id"],
        "status": "approved",
    }

    bad = asyncio.run(
        server.call_tool("propose", {"run_id": run, "key": "lr", "value": 0.1, "reason": "100x"})
    )
    assert bad.is_error is False
    assert bad.structured_content["accepted"] is False and bad.structured_content["sqlstate"] == "SE006"
    assert "max_rel_change is 3x" in bad.structured_content["message"]

    unknown = asyncio.run(
        server.call_tool("propose", {"run_id": run, "key": "secret", "value": 1, "reason": "x"})
    )
    assert unknown.structured_content["sqlstate"] == "SE003"

    hist = asyncio.run(server.call_tool("history", {"run_id": run, "limit": 10}))
    rows = hist.structured_content["result"]
    assert [r["status"] for r in rows] == ["approved"] and rows[0]["reason"] == "halve"


def test_mcp_run_state_for_missing_run(roles: Roles) -> None:
    assert "error" in AgentDB(roles.agent).run_state(999999999)


def test_coerce_value() -> None:
    assert coerce_value("0.5") == 0.5
    assert coerce_value("true") is True
    assert coerce_value("cosine") == "cosine"
    assert coerce_value('"cosine"') == "cosine"
    assert coerce_value(3) == 3


# ----- tuner -----


class FakeMessages:
    def __init__(self, answers: list[str], *, stop_reason: str = "end_turn") -> None:
        self.answers = answers
        self.calls: list[dict[str, Any]] = []
        self.stop_reason = stop_reason

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        text = self.answers[min(len(self.calls) - 1, len(self.answers) - 1)]
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)],
            stop_reason=self.stop_reason,
            usage=SimpleNamespace(input_tokens=1000, output_tokens=100),
        )


class FakeClient:
    def __init__(self, answers: list[str], **kw: Any) -> None:
        self.messages = FakeMessages(answers, **kw)


def test_tuner_proposes_through_the_database(roles: Roles, run: int, trainer: Any) -> None:
    answer = json.dumps(
        {
            "proposals": [
                {"key": "lr", "value": 0.0005, "reason": "loss flat for 100 steps"},
                {"key": "lr_scheduler", "value": "linear", "reason": "not in the enum"},
                {"key": "resolution", "value": 1280, "reason": "frozen key"},
            ],
            "note": "loss plateau",
        }
    )
    client = FakeClient([answer])
    tuner = tune.Tuner(AgentDB(roles.agent), client, model="claude-opus-5-5", max_usd=1.0)
    rounds = tuner.run(run, rounds=1, interval_s=0)
    assert len(rounds) == 1 and rounds[0].note == "loss plateau"
    statuses = [(r["key"], r["accepted"], r.get("sqlstate")) for r in rounds[0].results]
    assert statuses == [("lr", True, None), ("lr_scheduler", False, "SE005"), ("resolution", False, "SE004")]
    assert rounds[0].usd == pytest.approx(1000 * 4e-6 + 100 * 20e-6)

    call = client.messages.calls[0]
    assert call["model"] == "claude-opus-5-5" and call["system"] == tune.SYSTEM
    sent = json.loads(call["messages"][0]["content"])
    assert (
        sent["config"]["lr"] == 1e-3 and "policies" in sent and "note" not in sent["policies"]["resolution"]
    )
    row = trainer.execute("SELECT status FROM steerable.proposal_status WHERE run_id = %s", (run,)).fetchall()
    assert [r["status"] for r in row] == ["approved"]


def test_tuner_handles_garbage_and_refusals(roles: Roles, run: int) -> None:
    tuner = tune.Tuner(AgentDB(roles.agent), FakeClient(["not json at all"]))
    rnd = tuner.round(run)
    assert rnd is not None and rnd.proposals == [] and "unparseable" in rnd.note

    tuner = tune.Tuner(AgentDB(roles.agent), FakeClient(['{"proposals": "lr"}']))
    rnd = tuner.round(run)
    assert rnd is not None and rnd.proposals == []

    tuner = tune.Tuner(AgentDB(roles.agent), FakeClient(["ignored"], stop_reason="refusal"))
    rnd = tuner.round(run)
    assert rnd is not None and rnd.proposals == [] and "declined" in rnd.note


def test_tuner_spend_guard_and_finished_run(roles: Roles, run: int, trainer: Any) -> None:
    client = FakeClient(['{"proposals": [], "note": "fine"}'])
    tuner = tune.Tuner(AgentDB(roles.agent), client, max_usd=0.005)
    rounds = tuner.run(run, rounds=5, interval_s=0)
    assert len(rounds) == 1 and len(client.messages.calls) == 1  # second round blocked by the guard
    trainer.execute("SELECT steerable.finish(%s, 'finished')", (run,))
    assert tune.Tuner(AgentDB(roles.agent), FakeClient(["x"])).round(run) is None


def test_parse_decision_accepts_fenced_json() -> None:
    note, proposals = tune.parse_decision(
        '```json\n{"proposals": [{"key": "lr", "value": 1}], "note": "n"}\n```'
    )
    assert note == "n" and proposals == [{"key": "lr", "value": 1, "reason": "no reason given"}]
    _, many = tune.parse_decision(json.dumps({"proposals": [{"key": f"k{i}", "value": i} for i in range(9)]}))
    assert len(many) == 5
