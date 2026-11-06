"""Optional torch.distributed plumbing. Single-process when torch is absent or not initialised."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Comm:
    rank: int
    world_size: int
    broadcast: Callable[[Any], Any]
    all_gather: Callable[[Any], list[Any]]

    @property
    def is_leader(self) -> bool:
        return self.rank == 0


def _single() -> Comm:
    return Comm(rank=0, world_size=1, broadcast=lambda obj: obj, all_gather=lambda obj: [obj])


def detect() -> Comm:
    try:
        import torch.distributed as dist
    except ImportError:
        return _single()
    if not (dist.is_available() and dist.is_initialized()):
        return _single()

    def broadcast(obj: Any) -> Any:
        box = [obj]
        dist.broadcast_object_list(box, src=0)
        return box[0]

    def all_gather(obj: Any) -> list[Any]:
        out: list[Any] = [None] * dist.get_world_size()
        dist.all_gather_object(out, obj)
        return out

    return Comm(
        rank=dist.get_rank(), world_size=dist.get_world_size(), broadcast=broadcast, all_gather=all_gather
    )
