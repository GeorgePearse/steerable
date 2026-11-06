"""D-FINE adapter (github.com/Peterande/D-FINE, Apache-2.0; also fits RT-DETR and DEIM, which share the solver).

D-FINE is a repository, not a package: ``DetSolver.fit()`` calls ``train_one_epoch`` on
``self.train_dataloader`` once per epoch and ``evaluate`` after each one, with no callback surface.
The least invasive hook is the dataloader itself: ``SteerableLoader`` wraps it, calls
``sync(step)`` before yielding each batch and ``sync(step, at_validation=True)`` when the epoch's
iteration ends, which is exactly where ``fit()`` evaluates. ``steerable_solver`` returns a
``DetSolver`` subclass whose ``train()`` installs the wrapper and launches the run::

    from src.solver import DetSolver                 # inside the D-FINE repo
    from steerable.adapters.dfine import steerable_solver

    Solver = steerable_solver(DetSolver, dsn=TRAINER_DSN, run_id=RUN_ID)
    Solver(cfg).fit()

Steps are global optimizer steps (``epoch * len(loader) + i``, as D-FINE counts them). The LR
scheduler is epoch-based, so ``lr`` is rescaled through ``base_lrs`` and survives the next
``lr_scheduler.step()``. ``clip_max_norm`` is read by ``fit()`` once per epoch, so it is a
``validation``-boundary key. Nothing here imports D-FINE or torch at module load.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any

from steerable import torch as st
from steerable.client import Steerable
from steerable.types import Applier, Policy

POLICIES: list[Policy] = [
    Policy("lr", {"type": "number", "min": 1e-8, "max": 1e-1}, mode="auto", max_rel_change=3, min_steps_between=50, max_per_hour=6),
    Policy("weight_decay", {"type": "number", "min": 0, "max": 1}, mode="review", max_rel_change=10),
    Policy("loss_vfl", {"type": "number", "min": 0, "max": 50}, mode="auto", max_rel_change=3, max_per_hour=4),
    Policy("loss_bbox", {"type": "number", "min": 0, "max": 50}, mode="auto", max_rel_change=3, max_per_hour=4),
    Policy("loss_giou", {"type": "number", "min": 0, "max": 50}, mode="auto", max_rel_change=3, max_per_hour=4),
    Policy("loss_fgl", {"type": "number", "min": 0, "max": 50}, mode="auto", max_rel_change=3, max_per_hour=4),
    Policy("loss_ddf", {"type": "number", "min": 0, "max": 50}, mode="auto", max_rel_change=3, max_per_hour=4),
    Policy("ema_decay", {"type": "number", "min": 0.9, "max": 0.99999}, mode="review"),
    Policy("clip_max_norm", {"type": "number", "min": 0, "max": 100}, boundary="validation", mode="auto", max_per_hour=4),
    Policy("batch_size", {"type": "integer", "min": 1, "max": 512}, boundary="frozen",
           note="the DataLoader and its DistributedSampler are built once in train()"),
    Policy("epochs", {"type": "integer", "min": 1, "max": 10000}, boundary="frozen",
           note="MultiStepLR milestones, stop_epoch and the checkpoint cadence are sized from it"),
    Policy("stop_epoch", {"type": "integer", "min": 0, "max": 10000}, boundary="frozen",
           note="switches the multi-scale augmentation off and reloads best_stg1; a schedule, not a knob"),
    Policy("use_amp", {"type": "boolean"}, boundary="frozen",
           note="the GradScaler and autocast context are created at setup"),
    Policy("use_ema", {"type": "boolean"}, boundary="frozen",
           note="evaluation and checkpoints read the EMA module throughout the run"),
    Policy("eval_spatial_size", {"type": "integer", "min": 64, "max": 2048}, boundary="frozen",
           note="the evaluator and the exported model are fixed to it"),
    Policy("num_classes", {"type": "integer", "min": 1, "max": 10000}, boundary="frozen",
           note="sizes the classification head"),
]  # fmt: skip

_LOSS_KEYS = ("loss_vfl", "loss_bbox", "loss_giou", "loss_fgl", "loss_ddf", "loss_focal", "loss_mal")


def _ema_decay_applier(ema: Any | None) -> Applier:
    """D-FINE's ``ModelEMA`` captures ``decay`` inside ``decay_fn``; both must change together."""
    check = st.number("ema_decay", minimum=0.0, maximum=1.0)

    def apply(value: Any, *, dry_run: bool) -> None:
        decay = check(value)
        if ema is None:
            raise ValueError("ema_decay: this run has no EMA model")
        if dry_run:
            return
        ema.decay = decay
        warmups = getattr(ema, "warmups", 0)
        if warmups:
            ema.decay_fn = lambda x: decay * (1 - math.exp(-x / warmups))
        else:
            ema.decay_fn = lambda x: decay

    return apply


def bind(solver: Any) -> tuple[dict[str, Applier], dict[str, Any]]:
    """Appliers and launch config for a ``DetSolver`` after ``train()`` has built its objects."""
    cfg = solver.cfg
    optimizer = solver.optimizer
    weight_dict = solver.criterion.weight_dict
    ema = getattr(solver, "ema", None)
    appliers: dict[str, Applier] = {
        "lr": st.optimizer_lr(optimizer, scheduler=getattr(solver, "lr_scheduler", None)),
        "weight_decay": st.optimizer_weight_decay(optimizer),
        "ema_decay": _ema_decay_applier(ema),
        "clip_max_norm": st.attribute(cfg, "clip_max_norm", st.number("clip_max_norm", minimum=0.0)),
    }
    config: dict[str, Any] = {
        "lr": float(optimizer.param_groups[0]["lr"]),
        "weight_decay": float(optimizer.param_groups[0].get("weight_decay", 0.0)),
        "clip_max_norm": float(getattr(cfg, "clip_max_norm", 0.0) or 0.0),
        "epochs": int(cfg.epochs),
        "use_amp": bool(getattr(cfg, "use_amp", False)),
        "use_ema": ema is not None,
    }
    for key in _LOSS_KEYS:
        if key in weight_dict:
            appliers[key] = st.loss_weight(weight_dict, key)
            config[key] = float(weight_dict[key])
    if ema is not None:
        config["ema_decay"] = float(ema.decay)
    loader = getattr(solver, "train_dataloader", None)
    if loader is not None and getattr(loader, "batch_size", None) is not None:
        config["batch_size"] = int(loader.batch_size)
    collate = getattr(loader, "collate_fn", None)
    if collate is not None and hasattr(collate, "stop_epoch"):
        config["stop_epoch"] = int(collate.stop_epoch)
    for name in ("eval_spatial_size", "num_classes"):
        value = getattr(cfg, name, None)
        if isinstance(value, int):
            config[name] = value
        elif isinstance(value, list | tuple) and value:
            config[name] = int(value[0])
    return appliers, config


class SteerableLoader:
    """Wraps the train dataloader: ``sync`` before every batch, ``sync(at_validation=True)`` at epoch end."""

    def __init__(self, loader: Any, steer: Steerable, *, start_epoch: int = 0) -> None:
        self._loader = loader
        self._steer = steer
        self._epoch = start_epoch
        self._step = start_epoch * len(loader)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch
        self._step = epoch * len(self._loader)
        if hasattr(self._loader, "set_epoch"):
            self._loader.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self._loader)

    def __iter__(self) -> Iterator[Any]:
        for batch in self._loader:
            self._steer.sync(self._step)
            yield batch
            self._step += 1
        self._steer.sync(self._step, at_validation=True)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._loader, name)


def steerable_solver(
    base: type, *, dsn: str, run_id: int, poll_interval_s: float = 5.0, policies: list[Policy] | None = None
) -> type:
    """A ``DetSolver`` subclass that installs the loader wrapper and launches the run in ``train()``."""

    class SteerableDetSolver(base):  # type: ignore[misc]
        steer: Steerable | None = None

        def train(self) -> None:
            super().train()
            appliers, config = bind(self)
            self.steer = Steerable(dsn, run_id, appliers, poll_interval_s=poll_interval_s)
            self.steer.launch(config, POLICIES if policies is None else policies)
            start_epoch = max(0, int(getattr(self, "last_epoch", -1)) + 1)
            wrapped: Any = SteerableLoader(self.train_dataloader, self.steer, start_epoch=start_epoch)
            self.train_dataloader = wrapped

        def fit(self) -> None:
            try:
                super().fit()
            except BaseException:
                if self.steer is not None:
                    self.steer.finish("aborted")
                    self.steer.close()
                raise
            if self.steer is not None:
                self.steer.finish("finished")
                self.steer.close()

    SteerableDetSolver.__name__ = f"Steerable{base.__name__}"
    return SteerableDetSolver
