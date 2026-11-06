"""The human CLI, end to end against the database."""

from __future__ import annotations

import json
from typing import Any

import pytest

from conftest import Roles, create_run
from steerable import cli


def run_cli(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str]:
    rc = cli.main(list(argv))
    return rc, capsys.readouterr().out


def test_propose_approve_reject_freeze_state_history(
    roles: Roles, run: int, capsys: pytest.CaptureFixture[str]
) -> None:
    rc, out = run_cli(
        capsys,
        "--dsn",
        roles.agent,
        "propose",
        "--run",
        str(run),
        "weight_decay",
        "0.001",
        "--reason",
        "more reg",
    )
    assert rc == 0 and out.startswith("proposal ") and out.strip().endswith("pending")
    pid = int(out.split()[1])

    rc, out = run_cli(
        capsys, "--dsn", roles.agent, "propose", "--run", str(run), "lr", "0.5", "--reason", "x"
    )
    assert rc == 1 and out.startswith("rejected [SE005]")

    rc, out = run_cli(capsys, "--dsn", roles.admin, "approve", str(pid), "--note", "ok")
    assert rc == 0 and out.strip() == f"proposal {pid} approved"
    rc, out = run_cli(capsys, "--dsn", roles.admin, "reject", str(pid))
    assert rc == 1 and out.startswith("refused [SE010]")

    rc, out = run_cli(capsys, "--dsn", roles.admin, "state", "--run", str(run))
    state = json.loads(out)
    assert state["open_proposals"][0]["status"] == "approved"

    rc, out = run_cli(capsys, "--dsn", roles.admin, "freeze", "--run", str(run))
    assert rc == 0 and "agent_enabled = false" in out
    rc, out = run_cli(
        capsys, "--dsn", roles.agent, "propose", "--run", str(run), "lr", "0.0005", "--reason", "x"
    )
    assert rc == 1 and "[SE002]" in out
    rc, out = run_cli(capsys, "--dsn", roles.admin, "unfreeze", "--run", str(run))
    assert rc == 0

    rc, out = run_cli(capsys, "--dsn", roles.agent, "history", "--run", str(run))
    hist = json.loads(out)
    assert [h["status"] for h in hist] == ["approved"] and hist[0]["events"][0]["note"] == "ok"


def test_run_create_and_string_values(
    roles: Roles, capsys: pytest.CaptureFixture[str], trainer: Any, admin: Any
) -> None:
    rc, out = run_cli(
        capsys, "--dsn", roles.admin, "run", "create", "cli-made-run-" + str(abs(hash(roles.admin)) % 10**6)
    )
    assert rc == 0
    run_id = int(out.strip())
    row = admin.execute("SELECT status FROM steerable.runs WHERE id = %s", (run_id,)).fetchone()
    assert row["status"] == "created"

    run = create_run(admin, trainer)
    rc, out = run_cli(
        capsys, "--dsn", roles.agent, "propose", "--run", str(run), "lr_scheduler", "step", "--reason", "x"
    )
    assert rc == 0 and out.strip().endswith("approved")


def test_agent_cannot_use_admin_commands(roles: Roles, run: int, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(Exception, match="permission denied"):
        cli.main(["--dsn", roles.agent, "freeze", "--run", str(run)])
    rc, out = run_cli(capsys, "--dsn", roles.agent, "approve", "1")
    assert rc == 1 and "refused [42501]" in out
