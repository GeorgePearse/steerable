from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import psycopg
import pytest
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from steerable import db
from steerable.types import Policy

SUPERUSER_DSN = os.environ.get(
    "STEERABLE_TEST_DSN", "postgresql://postgres:postgres@localhost:5440/steerable"
)
PASSWORDS = {db.ADMIN: "admin-test", db.TRAINER: "trainer-test", db.AGENT: "agent-test"}


@dataclass(frozen=True)
class Roles:
    superuser: str
    admin: str
    trainer: str
    agent: str


def connect(dsn: str) -> psycopg.Connection[dict[str, Any]]:
    return psycopg.connect(dsn, row_factory=dict_row, autocommit=True)


@pytest.fixture(scope="session")
def roles() -> Roles:
    try:
        db.init(
            SUPERUSER_DSN,
            admin_password=PASSWORDS[db.ADMIN],
            trainer_password=PASSWORDS[db.TRAINER],
            agent_password=PASSWORDS[db.AGENT],
            reset=True,
        )
    except psycopg.OperationalError as exc:
        pytest.skip(f"no test Postgres at {SUPERUSER_DSN}: {exc}")
    return Roles(
        superuser=SUPERUSER_DSN,
        admin=db.role_dsn(SUPERUSER_DSN, db.ADMIN, PASSWORDS[db.ADMIN]),
        trainer=db.role_dsn(SUPERUSER_DSN, db.TRAINER, PASSWORDS[db.TRAINER]),
        agent=db.role_dsn(SUPERUSER_DSN, db.AGENT, PASSWORDS[db.AGENT]),
    )


@pytest.fixture
def admin(roles: Roles) -> Iterator[psycopg.Connection[dict[str, Any]]]:
    with connect(roles.admin) as conn:
        yield conn


@pytest.fixture
def trainer(roles: Roles) -> Iterator[psycopg.Connection[dict[str, Any]]]:
    with connect(roles.trainer) as conn:
        yield conn


@pytest.fixture
def agent(roles: Roles) -> Iterator[psycopg.Connection[dict[str, Any]]]:
    with connect(roles.agent) as conn:
        yield conn


DEFAULT_POLICIES = [
    Policy("lr", {"type": "number", "min": 1e-7, "max": 1e-1}, mode="auto", max_rel_change=3, max_per_hour=5),
    Policy("weight_decay", {"type": "number", "min": 0, "max": 1}, mode="review"),
    Policy("grad_accum_steps", {"type": "integer", "min": 1, "max": 64}, mode="auto", min_steps_between=100),
    Policy("use_amp", {"type": "boolean"}, mode="auto"),
    Policy("lr_scheduler", {"type": "enum", "values": ["cosine", "step", "constant"]}, mode="auto"),
    Policy("batch_size", {"type": "integer", "min": 1, "max": 512}, boundary="validation", mode="auto"),
    Policy(
        "resolution", {"type": "integer", "min": 64, "max": 2048}, boundary="frozen", note="sizes the head"
    ),
]
LAUNCH_CONFIG = {
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "grad_accum_steps": 1,
    "use_amp": False,
    "lr_scheduler": "cosine",
    "batch_size": 16,
    "resolution": 640,
    "model": "toy",
}


def create_run(
    admin: psycopg.Connection[dict[str, Any]],
    trainer: psycopg.Connection[dict[str, Any]],
    *,
    launch: bool = True,
    config: dict[str, Any] | None = None,
    policies: list[Policy] | None = None,
    max_proposals_per_hour: int = 30,
) -> int:
    row = admin.execute(
        "INSERT INTO steerable.runs (name, max_proposals_per_hour) VALUES (%s, %s) RETURNING id",
        (f"t-{uuid.uuid4().hex[:12]}", max_proposals_per_hour),
    ).fetchone()
    assert row is not None
    run_id = int(row["id"])
    if launch:
        trainer.execute(
            "SELECT steerable.launch(%s, %s, %s)",
            (
                run_id,
                Jsonb(config if config is not None else LAUNCH_CONFIG),
                Jsonb([p.to_json() for p in (policies if policies is not None else DEFAULT_POLICIES)]),
            ),
        )
    return run_id


@pytest.fixture
def run(admin: psycopg.Connection[dict[str, Any]], trainer: psycopg.Connection[dict[str, Any]]) -> int:
    return create_run(admin, trainer)
