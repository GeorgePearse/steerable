from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Literal, Protocol

Boundary = Literal["step", "validation", "frozen"]
Mode = Literal["auto", "review", "frozen"]


class Applier(Protocol):
    """Applies a typed value to the live training process.

    Called once with ``dry_run=True`` for every proposal in a batch, on every rank, before any
    value is applied. Raise to reject; the exception text is recorded as ``trainer_rejected``.
    """

    def __call__(self, value: Any, *, dry_run: bool) -> None: ...


@dataclass(frozen=True)
class Policy:
    """What may change for one key, when, by how much and how often."""

    key: str
    schema: dict[str, Any]
    boundary: Boundary = "step"
    mode: Mode = "review"
    max_rel_change: float | None = None
    min_steps_between: int = 0
    max_per_hour: int = 6
    note: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Proposal:
    """One proposed change, as the trainer sees it."""

    id: int
    run_id: int
    key: str
    value: Any
    previous: Any
    actor_kind: str
    reason: str
    status: str
    boundary: Boundary
    created_at: datetime = field(compare=False)
