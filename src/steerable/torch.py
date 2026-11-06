"""Generic appliers for torch training loops. Duck-typed: nothing here imports torch.

Each factory returns an ``Applier``. With ``dry_run=True`` it validates and touches nothing;
with ``dry_run=False`` it applies. Validation errors raise ``ValueError`` whose text becomes
the ``trainer_rejected`` note.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, MutableMapping
from typing import Any

from steerable.types import Applier

Check = Callable[[Any], Any]


def number(
    name: str, *, minimum: float = 0.0, maximum: float = math.inf, exclusive_min: bool = False
) -> Check:
    def check(value: Any) -> float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{name}: expected a number, got {type(value).__name__}")
        v = float(value)
        if not math.isfinite(v):
            raise ValueError(f"{name}: must be finite")
        if v < minimum or (exclusive_min and v == minimum):
            raise ValueError(f"{name}: {v} is below the minimum {minimum}")
        if v > maximum:
            raise ValueError(f"{name}: {v} is above the maximum {maximum}")
        return v

    return check


def integer(name: str, *, minimum: int = 0, maximum: int = 2**31 - 1) -> Check:
    def check(value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int | float) or value != int(value):
            raise ValueError(f"{name}: expected an integer, got {value!r}")
        v = int(value)
        if v < minimum or v > maximum:
            raise ValueError(f"{name}: {v} is outside [{minimum}, {maximum}]")
        return v

    return check


def boolean(name: str) -> Check:
    def check(value: Any) -> bool:
        if not isinstance(value, bool):
            raise ValueError(f"{name}: expected a boolean, got {value!r}")
        return value

    return check


def value_only(check: Check) -> Applier:
    """Accept a key the loop reads itself through ``Steerable.value(key)``; nothing to mutate."""

    def apply(value: Any, *, dry_run: bool) -> None:
        check(value)

    return apply


def attribute(target: object, attr: str, check: Check) -> Applier:
    """Set ``target.attr`` to the checked value."""
    if not hasattr(target, attr):
        raise AttributeError(f"{type(target).__name__} has no attribute {attr}")

    def apply(value: Any, *, dry_run: bool) -> None:
        v = check(value)
        if not dry_run:
            setattr(target, attr, v)

    return apply


def optimizer_lr(
    optimizer: Any,
    *,
    group: int | None = None,
    scheduler: Any | None = None,
    maximum: float = 10.0,
) -> Applier:
    """Set the learning rate of one param group or all of them.

    The value becomes the group's current ``lr``. When a scheduler is passed, its ``base_lrs``
    are rescaled by the same factor so later ``scheduler.step()`` calls continue the schedule
    from the new rate instead of reverting it.
    """
    check = number("lr", minimum=0.0, maximum=maximum)

    def groups() -> list[MutableMapping[str, Any]]:
        all_groups = list(optimizer.param_groups)
        if group is None:
            return all_groups
        if group < 0 or group >= len(all_groups):
            raise ValueError(f"lr: optimizer has {len(all_groups)} param groups; no group {group}")
        return [all_groups[group]]

    def apply(value: Any, *, dry_run: bool) -> None:
        lr = check(value)
        targets = groups()
        if dry_run:
            return
        for g in targets:
            if scheduler is not None:
                i = optimizer.param_groups.index(g)
                old = g["lr"]
                if old > 0 and scheduler.base_lrs[i] > 0:
                    scheduler.base_lrs[i] = scheduler.base_lrs[i] * (lr / old)
                else:
                    scheduler.base_lrs[i] = lr
            g["lr"] = lr

    return apply


def optimizer_weight_decay(optimizer: Any, *, group: int | None = None, maximum: float = 1.0) -> Applier:
    check = number("weight_decay", minimum=0.0, maximum=maximum)

    def apply(value: Any, *, dry_run: bool) -> None:
        wd = check(value)
        all_groups = list(optimizer.param_groups)
        if group is not None and (group < 0 or group >= len(all_groups)):
            raise ValueError(f"weight_decay: optimizer has {len(all_groups)} param groups; no group {group}")
        if dry_run:
            return
        for g in all_groups if group is None else [all_groups[group]]:
            g["weight_decay"] = wd

    return apply


def ema_decay(ema: Any | None, *, attr: str = "decay") -> Applier:
    """Set the decay of an EMA wrapper (``ModelEma.decay`` or similar). Rejects when there is no EMA."""
    check = number("ema_decay", minimum=0.0, maximum=1.0)

    def apply(value: Any, *, dry_run: bool) -> None:
        v = check(value)
        if ema is None:
            raise ValueError("ema_decay: this run has no EMA model")
        if not hasattr(ema, attr):
            raise ValueError(f"ema_decay: EMA object has no attribute {attr}")
        if not dry_run:
            setattr(ema, attr, v)

    return apply


_COPY_SUFFIX = r"(_(\d+|enc|enc_\d+|dn|dn_\d+|pre|aux_\d+))?"


def loss_weight(weight_dict: MutableMapping[str, float], name: str, *, maximum: float = 1000.0) -> Applier:
    """Set a loss coefficient in a DETR-style ``weight_dict``, including its per-layer copies.

    ``loss_giou`` also reaches ``loss_giou_0`` .. ``loss_giou_5``, ``loss_giou_enc`` and
    ``loss_giou_dn_*`` when present, so the change applies to every decoder layer's auxiliary
    loss and not only the last one.
    """
    pattern = re.compile(rf"^{re.escape(name)}{_COPY_SUFFIX}$")
    check = number(name, minimum=0.0, maximum=maximum)

    def keys() -> list[str]:
        found = [k for k in weight_dict if pattern.match(k)]
        if not found:
            raise ValueError(f"{name}: no matching key in weight_dict (have {sorted(weight_dict)[:8]} ...)")
        return found

    def apply(value: Any, *, dry_run: bool) -> None:
        w = check(value)
        targets = keys()
        if dry_run:
            return
        for k in targets:
            weight_dict[k] = w

    return apply
