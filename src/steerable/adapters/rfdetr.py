"""RF-DETR adapter (pip ``rfdetr`` >= 1.11, Roboflow, Apache-2.0).

RF-DETR 1.11 trains through PyTorch Lightning: ``RFDETR.train()`` builds a ``Trainer`` with
``rfdetr.training.trainer.build_trainer`` and calls ``fit`` at once. The least invasive hook is
therefore a Lightning ``Callback``. ``train()`` has no callback parameter, so attach it through
``build_trainer``::

    from steerable.adapters.rfdetr import SteerableCallback, attach

    model = RFDETRMedium()
    with attach(SteerableCallback(dsn, run_id)):
        model.train(dataset_dir="...", epochs=20)

``attach`` wraps ``build_trainer`` for the duration of the block and appends the callback to the
trainer RF-DETR built, leaving its own EMA, checkpoint and evaluation callbacks in place.

Steps are Lightning's ``global_step`` (optimizer steps). ``step`` keys apply before the next
training batch; ``validation`` keys apply when a validation epoch starts. Nothing here imports
rfdetr or pytorch_lightning at module load.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

from steerable import torch as st
from steerable.client import Steerable
from steerable.types import Applier, Policy

POLICIES: list[Policy] = [
    Policy("lr", {"type": "number", "min": 1e-8, "max": 1e-1}, mode="auto", max_rel_change=3, min_steps_between=50, max_per_hour=6),
    Policy("weight_decay", {"type": "number", "min": 0, "max": 1}, mode="review", max_rel_change=10),
    Policy("cls_loss_coef", {"type": "number", "min": 0, "max": 50}, mode="auto", max_rel_change=3, max_per_hour=4),
    Policy("bbox_loss_coef", {"type": "number", "min": 0, "max": 50}, mode="auto", max_rel_change=3, max_per_hour=4),
    Policy("giou_loss_coef", {"type": "number", "min": 0, "max": 50}, mode="auto", max_rel_change=3, max_per_hour=4),
    Policy("ema_decay", {"type": "number", "min": 0.9, "max": 0.99999}, mode="review"),
    Policy("clip_max_norm", {"type": "number", "min": 0, "max": 100}, mode="auto", max_per_hour=4),
    Policy("resolution", {"type": "integer", "min": 64, "max": 2048}, boundary="frozen",
           note="sizes the positional encoding and every multi-scale target; the model is built for it"),
    Policy("batch_size", {"type": "integer", "min": 1, "max": 512}, boundary="frozen",
           note="Lightning owns the dataloader for the whole fit; a rebuild needs a new run"),
    Policy("grad_accum_steps", {"type": "integer", "min": 1, "max": 64}, boundary="frozen",
           note="Lightning's accumulation window is fixed at Trainer construction"),
    Policy("epochs", {"type": "integer", "min": 1, "max": 10000}, boundary="frozen",
           note="the cosine schedule, warmup and checkpoint cadence are sized from it at launch"),
    Policy("use_ema", {"type": "boolean"}, boundary="frozen",
           note="EMA weights feed evaluation, best-checkpoint selection and early stopping throughout"),
    Policy("amp_dtype", {"type": "enum", "values": ["bf16", "fp16", "fp32"]}, boundary="frozen",
           note="precision is fixed in the Trainer and the optimizer state is laid out for it"),
    Policy("lr_scheduler", {"type": "enum", "values": ["step", "cosine"]}, boundary="frozen",
           note="the scheduler object is constructed once in configure_optimizers"),
    Policy("multi_scale", {"type": "enum", "values": ["off", "per_batch"]}, boundary="frozen",
           note="changes the augmentation distribution the loss landscape was tuned for"),
]  # fmt: skip

_LOSS_KEYS = {"cls_loss_coef": "loss_ce", "bbox_loss_coef": "loss_bbox", "giou_loss_coef": "loss_giou"}


def _ema_callback(trainer: Any) -> Any | None:
    for cb in getattr(trainer, "callbacks", []):
        if type(cb).__name__ == "RFDETREMACallback":
            return cb
    return None


def bind_appliers(trainer: Any, pl_module: Any) -> dict[str, Applier]:
    """Appliers for a fitted RF-DETR ``Trainer`` and its ``RFDETRModelModule``."""
    optimizer = trainer.optimizers[0]
    schedulers = getattr(trainer, "lr_scheduler_configs", None) or []
    scheduler = schedulers[0].scheduler if schedulers else None
    weight_dict = pl_module.criterion.weight_dict
    appliers: dict[str, Applier] = {
        "lr": st.optimizer_lr(optimizer, scheduler=scheduler),
        "weight_decay": st.optimizer_weight_decay(optimizer),
        "clip_max_norm": st.attribute(trainer, "gradient_clip_val", st.number("clip_max_norm", minimum=0.0)),
    }
    for key, loss in _LOSS_KEYS.items():
        if any(k == loss or k.startswith(loss + "_") for k in weight_dict):
            appliers[key] = st.loss_weight(weight_dict, loss)
    appliers["ema_decay"] = st.ema_decay(_ema_callback(trainer), attr="_decay")
    return appliers


def launch_config(train_config: Any, model_config: Any) -> dict[str, Any]:
    """The launch values for every policy key, read from RF-DETR's configs."""
    tc, mc = train_config, model_config
    return {
        "lr": float(tc.lr),
        "weight_decay": float(tc.weight_decay),
        "cls_loss_coef": float(tc.cls_loss_coef),
        "bbox_loss_coef": float(getattr(tc, "bbox_loss_coef", 5.0)),
        "giou_loss_coef": float(getattr(tc, "giou_loss_coef", 2.0)),
        "ema_decay": float(tc.ema_decay),
        "clip_max_norm": float(tc.clip_max_norm),
        "resolution": int(mc.resolution),
        "batch_size": tc.batch_size if isinstance(tc.batch_size, int) else str(tc.batch_size),
        "grad_accum_steps": int(tc.grad_accum_steps),
        "epochs": int(tc.epochs),
        "use_ema": bool(tc.use_ema),
        "amp_dtype": str(getattr(tc, "amp_dtype", "bf16")),
        "lr_scheduler": str(tc.lr_scheduler) if isinstance(tc.lr_scheduler, str) else "custom",
        "multi_scale": str(getattr(tc.multi_scale, "value", tc.multi_scale)).lower(),
    }


def _callback_base() -> type:
    try:
        from pytorch_lightning import Callback

        base: type = Callback
    except ImportError:

        class _Callback:  # duck-typed stand-in so the adapter can be unit-tested without Lightning
            pass

        base = _Callback
    return base


class SteerableCallback(_callback_base()):  # type: ignore[misc]
    """Lightning callback: launch on fit start, sync every batch, validation keys at validation start."""

    _VAL_METRICS = ("val/map", "val/map50", "val/loss", "val/ema_map", "val/ema_map50")

    def __init__(
        self, dsn: str, run_id: int, *, poll_interval_s: float = 5.0, policies: list[Policy] | None = None
    ) -> None:
        super().__init__()
        self.dsn = dsn
        self.run_id = run_id
        self.poll_interval_s = poll_interval_s
        self.policies = POLICIES if policies is None else policies
        self.steer: Steerable | None = None

    def on_fit_start(self, trainer: Any, pl_module: Any) -> None:
        appliers = bind_appliers(trainer, pl_module)
        self.steer = Steerable(self.dsn, self.run_id, appliers, poll_interval_s=self.poll_interval_s)
        self.steer.launch(launch_config(pl_module.train_config, pl_module.model_config), self.policies)

    def on_train_batch_start(self, trainer: Any, pl_module: Any, batch: Any, batch_idx: int) -> None:
        if self.steer is not None:
            self.steer.sync(int(trainer.global_step))

    def on_train_batch_end(
        self, trainer: Any, pl_module: Any, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        if self.steer is None:
            return
        loss = trainer.callback_metrics.get("train/loss")
        if loss is not None:
            self.steer.log_metric(int(trainer.global_step), "train_loss", float(loss))

    def on_validation_start(self, trainer: Any, pl_module: Any) -> None:
        if self.steer is not None and not getattr(trainer, "sanity_checking", False):
            self.steer.sync(int(trainer.global_step), at_validation=True)

    def on_validation_end(self, trainer: Any, pl_module: Any) -> None:
        if self.steer is None or getattr(trainer, "sanity_checking", False):
            return
        for name in self._VAL_METRICS:
            value = trainer.callback_metrics.get(name)
            if value is not None:
                self.steer.log_metric(int(trainer.global_step), name.replace("/", "_"), float(value))

    def on_fit_end(self, trainer: Any, pl_module: Any) -> None:
        if self.steer is not None:
            self.steer.finish("finished")
            self.steer.close()
            self.steer = None

    def on_exception(self, trainer: Any, pl_module: Any, exception: BaseException) -> None:
        if self.steer is not None:
            self.steer.finish("aborted")
            self.steer.close()
            self.steer = None


@contextlib.contextmanager
def attach(callback: Any, module: Any | None = None) -> Iterator[None]:
    """Append ``callback`` to every Trainer that ``rfdetr.detr.build_trainer`` returns inside the block."""
    if module is None:
        import importlib

        module = importlib.import_module("rfdetr.detr")
    target: Any = module
    original = target.build_trainer

    def build_trainer(*args: Any, **kwargs: Any) -> Any:
        trainer = original(*args, **kwargs)
        trainer.callbacks.append(callback)
        return trainer

    target.build_trainer = build_trainer
    try:
        yield
    finally:
        target.build_trainer = original
