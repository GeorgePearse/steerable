"""RF-DETR and D-FINE adapters against lightweight fakes: no rfdetr, Lightning, D-FINE or GPU."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest

from conftest import Roles
from steerable.adapters import dfine, rfdetr


class FakeOptimizer:
    def __init__(self, lr: float = 1e-4, wd: float = 1e-4) -> None:
        self.param_groups = [{"lr": lr, "weight_decay": wd}, {"lr": lr * 1.5, "weight_decay": wd}]


class FakeScheduler:
    def __init__(self, opt: FakeOptimizer) -> None:
        self.base_lrs = [g["lr"] for g in opt.param_groups]


class RFDETREMACallback:  # name matters: the adapter finds it by class name
    def __init__(self) -> None:
        self._decay = 0.993


def rfdetr_weight_dict() -> dict[str, float]:
    base = {"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0}
    out = dict(base)
    for i in range(3):
        out.update({f"{k}_{i}": v for k, v in base.items()})
    out.update({f"{k}_enc": v for k, v in base.items()})
    return out


def fake_lightning(weight_dict: dict[str, float] | None = None) -> tuple[Any, Any]:
    opt = FakeOptimizer()
    trainer = SimpleNamespace(
        optimizers=[opt],
        lr_scheduler_configs=[SimpleNamespace(scheduler=FakeScheduler(opt))],
        gradient_clip_val=0.1,
        callbacks=[RFDETREMACallback()],
        global_step=0,
        callback_metrics={},
        sanity_checking=False,
    )
    module = SimpleNamespace(
        criterion=SimpleNamespace(weight_dict=weight_dict if weight_dict is not None else rfdetr_weight_dict()),
        train_config=SimpleNamespace(
            lr=1e-4, weight_decay=1e-4, cls_loss_coef=1.0, bbox_loss_coef=5.0, giou_loss_coef=2.0, ema_decay=0.993,
            clip_max_norm=0.1, batch_size=4, grad_accum_steps=4, epochs=50, use_ema=True, amp_dtype="bf16",
            lr_scheduler="step", multi_scale=SimpleNamespace(value="per_batch"),
        ),
        model_config=SimpleNamespace(resolution=576),
    )  # fmt: skip
    return trainer, module


# ----- RF-DETR -----


def test_rfdetr_policies_cover_launch_config_keys() -> None:
    _, module = fake_lightning()
    config = rfdetr.launch_config(module.train_config, module.model_config)
    assert set(config) == {p.key for p in rfdetr.POLICIES}
    frozen = {p.key for p in rfdetr.POLICIES if p.boundary == "frozen"}
    assert {"resolution", "batch_size", "grad_accum_steps", "epochs", "use_ema"} <= frozen
    assert all(p.note for p in rfdetr.POLICIES if p.boundary == "frozen")


def test_rfdetr_appliers_reach_the_right_objects() -> None:
    trainer, module = fake_lightning()
    appliers = rfdetr.bind_appliers(trainer, module)
    assert set(appliers) == {
        "lr",
        "weight_decay",
        "clip_max_norm",
        "cls_loss_coef",
        "bbox_loss_coef",
        "giou_loss_coef",
        "ema_decay",
    }

    appliers["lr"](5e-5, dry_run=False)
    assert [g["lr"] for g in trainer.optimizers[0].param_groups] == [5e-5, 5e-5]
    assert trainer.lr_scheduler_configs[0].scheduler.base_lrs[0] == pytest.approx(5e-5)

    appliers["giou_loss_coef"](4.0, dry_run=False)
    wd = module.criterion.weight_dict
    assert {k: v for k, v in wd.items() if k.startswith("loss_giou")} == dict.fromkeys(
        ["loss_giou", "loss_giou_0", "loss_giou_1", "loss_giou_2", "loss_giou_enc"], 4.0
    )
    assert wd["loss_bbox_0"] == 5.0

    appliers["clip_max_norm"](0.5, dry_run=False)
    assert trainer.gradient_clip_val == 0.5
    appliers["ema_decay"](0.999, dry_run=False)
    assert trainer.callbacks[0]._decay == 0.999
    with pytest.raises(ValueError):
        appliers["ema_decay"](1.5, dry_run=True)


def test_rfdetr_without_ema_rejects_ema_decay() -> None:
    trainer, module = fake_lightning()
    trainer.callbacks = []
    with pytest.raises(ValueError, match="no EMA"):
        rfdetr.bind_appliers(trainer, module)["ema_decay"](0.99, dry_run=True)


def test_rfdetr_callback_lifecycle_against_database(
    roles: Roles, admin: Any, trainer: Any, agent: Any
) -> None:
    from conftest import create_run

    run = create_run(admin, trainer, launch=False)
    pl_trainer, module = fake_lightning()
    cb = rfdetr.SteerableCallback(roles.trainer, run, poll_interval_s=0.1)
    cb.on_fit_start(pl_trainer, module)
    assert cb.steer is not None and cb.steer.value("lr") == 1e-4
    row = trainer.execute("SELECT status FROM steerable.runs WHERE id = %s", (run,)).fetchone()
    assert row["status"] == "running"

    agent.execute("SELECT * FROM steerable.propose(%s, 'lr', '0.00005', 'halve')", (run,))
    assert cb.steer.wait_for_proposals(5)
    pl_trainer.global_step = 120
    cb.on_train_batch_start(pl_trainer, module, None, 0)
    assert pl_trainer.optimizers[0].param_groups[0]["lr"] == 5e-5
    pl_trainer.callback_metrics = {"train/loss": 3.25}
    cb.on_train_batch_end(pl_trainer, module, None, None, 0)
    pl_trainer.callback_metrics = {"val/map": 0.41}
    cb.on_validation_end(pl_trainer, module)
    cb.on_fit_end(pl_trainer, module)
    assert cb.steer is None

    hist = trainer.execute(
        "SELECT status, status_step FROM steerable.proposal_status WHERE run_id = %s", (run,)
    ).fetchall()
    assert [(h["status"], h["status_step"]) for h in hist] == [("applied", 120)]
    metrics = trainer.execute(
        "SELECT name, value FROM steerable.metrics WHERE run_id = %s ORDER BY name", (run,)
    ).fetchall()
    assert [(m["name"], m["value"]) for m in metrics] == [("train_loss", 3.25), ("val_map", 0.41)]
    row = trainer.execute("SELECT status FROM steerable.runs WHERE id = %s", (run,)).fetchone()
    assert row["status"] == "finished"


def test_rfdetr_attach_appends_callback_to_built_trainer() -> None:
    built = SimpleNamespace(callbacks=["ema"])
    module = SimpleNamespace(build_trainer=lambda *a, **k: built)
    original = module.build_trainer
    with rfdetr.attach("steer", module):
        assert module.build_trainer(1, x=2) is built
        assert built.callbacks == ["ema", "steer"]
    assert module.build_trainer is original


# ----- D-FINE -----


class FakeEMA:
    def __init__(self) -> None:
        self.decay = 0.9999
        self.warmups = 2000
        self.decay_fn = lambda x: 0.9999 * (1 - math.exp(-x / 2000))


class FakeLoader:
    def __init__(self, batches: list[Any]) -> None:
        self.batches = batches
        self.batch_size = 8
        self.collate_fn = SimpleNamespace(stop_epoch=72)
        self.epochs_set: list[int] = []

    def set_epoch(self, epoch: int) -> None:
        self.epochs_set.append(epoch)

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self) -> Any:
        return iter(self.batches)


class FakeSolver:
    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.last_epoch = -1
        self.trained = False

    def train(self) -> None:
        self.trained = True
        self.optimizer = FakeOptimizer(lr=2.5e-4)
        self.lr_scheduler = FakeScheduler(self.optimizer)
        self.criterion = SimpleNamespace(
            weight_dict={
                "loss_vfl": 1.0,
                "loss_bbox": 5.0,
                "loss_giou": 2.0,
                "loss_fgl": 0.15,
                "loss_ddf": 1.5,
            }
        )
        self.ema = FakeEMA()
        self.train_dataloader = FakeLoader([("b", 0), ("b", 1), ("b", 2)])

    def fit(self) -> None:
        self.train()
        for epoch in range(2):
            self.train_dataloader.set_epoch(epoch)
            for _ in self.train_dataloader:
                pass


def fake_cfg() -> Any:
    return SimpleNamespace(
        clip_max_norm=0.1, epochs=72, use_amp=True, eval_spatial_size=[640, 640], num_classes=80
    )


def test_dfine_bind_builds_appliers_and_config() -> None:
    solver = FakeSolver(fake_cfg())
    solver.train()
    appliers, config = dfine.bind(solver)
    assert set(appliers) == {
        "lr",
        "weight_decay",
        "ema_decay",
        "clip_max_norm",
        "loss_vfl",
        "loss_bbox",
        "loss_giou",
        "loss_fgl",
        "loss_ddf",
    }
    assert config == {
        "lr": 2.5e-4, "weight_decay": 1e-4, "clip_max_norm": 0.1, "epochs": 72, "use_amp": True, "use_ema": True,
        "loss_vfl": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0, "loss_fgl": 0.15, "loss_ddf": 1.5,
        "ema_decay": 0.9999, "batch_size": 8, "stop_epoch": 72, "eval_spatial_size": 640, "num_classes": 80,
    }  # fmt: skip
    assert set(config) <= {p.key for p in dfine.POLICIES}

    appliers["ema_decay"](0.999, dry_run=True)
    assert solver.ema.decay == 0.9999
    appliers["ema_decay"](0.999, dry_run=False)
    assert solver.ema.decay == 0.999 and solver.ema.decay_fn(10**9) == pytest.approx(0.999)
    appliers["clip_max_norm"](1.0, dry_run=False)
    assert solver.cfg.clip_max_norm == 1.0
    appliers["lr"](1e-4, dry_run=False)
    assert solver.lr_scheduler.base_lrs[0] == pytest.approx(1e-4)


def test_dfine_loader_syncs_each_batch_and_at_epoch_end() -> None:
    calls: list[tuple[int, bool]] = []
    steer = SimpleNamespace(sync=lambda step, at_validation=False: calls.append((step, at_validation)))
    loader = FakeLoader(["a", "b", "c"])
    wrapped = dfine.SteerableLoader(loader, steer, start_epoch=0)  # type: ignore[arg-type]
    assert len(wrapped) == 3 and wrapped.batch_size == 8 and wrapped.collate_fn.stop_epoch == 72
    wrapped.set_epoch(1)
    assert list(wrapped) == ["a", "b", "c"] and loader.epochs_set == [1]
    assert calls == [(3, False), (4, False), (5, False), (6, True)]


def test_dfine_steerable_solver_end_to_end(roles: Roles, admin: Any, trainer: Any, agent: Any) -> None:
    from conftest import create_run

    run = create_run(admin, trainer, launch=False)
    Solver = dfine.steerable_solver(FakeSolver, dsn=roles.trainer, run_id=run, poll_interval_s=0.1)
    assert Solver.__name__ == "SteerableFakeSolver"
    solver = Solver(fake_cfg())

    class Hooked(Solver):  # type: ignore[misc, valid-type]
        def train(self) -> None:
            super().train()
            agent.execute("SELECT * FROM steerable.propose(%s, 'loss_giou', '4', 'more giou')", (run,))
            agent.execute(
                "SELECT * FROM steerable.propose(%s, 'clip_max_norm', '0.5', 'looser clip')", (run,)
            )
            assert self.steer is not None and self.steer.wait_for_proposals(5)
            import time

            time.sleep(0.3)

    solver = Hooked(fake_cfg())
    solver.fit()
    assert solver.criterion.weight_dict["loss_giou"] == 4.0
    assert solver.cfg.clip_max_norm == 0.5
    hist = trainer.execute(
        "SELECT key, status, status_step FROM steerable.proposal_status WHERE run_id = %s ORDER BY proposal_id",
        (run,),
    ).fetchall()
    assert [(h["key"], h["status"], h["status_step"]) for h in hist] == [
        ("loss_giou", "applied", 0),
        ("clip_max_norm", "applied", 3),
    ]
    row = trainer.execute(
        "SELECT status, launch_config->>'lr' AS lr FROM steerable.runs WHERE id = %s", (run,)
    ).fetchone()
    assert row["status"] == "finished" and float(row["lr"]) == 2.5e-4
