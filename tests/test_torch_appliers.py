"""Generic appliers with lightweight fakes: no torch import, no GPU."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from steerable import torch as st


class FakeOptimizer:
    def __init__(self, lrs: list[float]) -> None:
        self.param_groups = [{"lr": lr, "weight_decay": 0.01} for lr in lrs]


class FakeScheduler:
    def __init__(self, opt: FakeOptimizer) -> None:
        self.base_lrs = [g["lr"] for g in opt.param_groups]


def test_optimizer_lr_all_groups_dry_run_then_apply() -> None:
    opt = FakeOptimizer([1e-3, 1e-4])
    apply = st.optimizer_lr(opt)
    apply(5e-4, dry_run=True)
    assert [g["lr"] for g in opt.param_groups] == [1e-3, 1e-4]
    apply(5e-4, dry_run=False)
    assert [g["lr"] for g in opt.param_groups] == [5e-4, 5e-4]


def test_optimizer_lr_single_group_and_bad_group() -> None:
    opt = FakeOptimizer([1e-3, 1e-4])
    st.optimizer_lr(opt, group=1)(2e-4, dry_run=False)
    assert [g["lr"] for g in opt.param_groups] == [1e-3, 2e-4]
    with pytest.raises(ValueError, match="no group 5"):
        st.optimizer_lr(opt, group=5)(1e-3, dry_run=True)


def test_optimizer_lr_rescales_scheduler_base() -> None:
    opt = FakeOptimizer([1e-3])
    sched = FakeScheduler(opt)
    opt.param_groups[0]["lr"] = 5e-4  # schedule is at factor 0.5
    st.optimizer_lr(opt, scheduler=sched)(1e-4, dry_run=False)
    assert opt.param_groups[0]["lr"] == 1e-4
    assert sched.base_lrs[0] == pytest.approx(2e-4)


@pytest.mark.parametrize("bad", ["0.001", True, None, [1e-3], float("nan"), -1.0, 11.0])
def test_optimizer_lr_rejects_non_numbers_and_out_of_range(bad: object) -> None:
    opt = FakeOptimizer([1e-3])
    with pytest.raises(ValueError):
        st.optimizer_lr(opt)(bad, dry_run=True)
    assert opt.param_groups[0]["lr"] == 1e-3


def test_weight_decay() -> None:
    opt = FakeOptimizer([1e-3, 1e-3])
    st.optimizer_weight_decay(opt)(0.05, dry_run=False)
    assert [g["weight_decay"] for g in opt.param_groups] == [0.05, 0.05]
    with pytest.raises(ValueError):
        st.optimizer_weight_decay(opt)(2.0, dry_run=True)


def test_ema_decay_requires_an_ema() -> None:
    with pytest.raises(ValueError, match="no EMA"):
        st.ema_decay(None)(0.999, dry_run=True)
    ema = SimpleNamespace(decay=0.9998)
    st.ema_decay(ema)(0.999, dry_run=True)
    assert ema.decay == 0.9998
    st.ema_decay(ema)(0.999, dry_run=False)
    assert ema.decay == 0.999
    with pytest.raises(ValueError):
        st.ema_decay(ema)(1.5, dry_run=True)


def test_loss_weight_reaches_per_layer_and_encoder_copies() -> None:
    wd = {
        "loss_giou": 2.0,
        "loss_giou_0": 2.0,
        "loss_giou_1": 2.0,
        "loss_giou_enc": 2.0,
        "loss_giou_dn_0": 2.0,
        "loss_bbox": 5.0,
        "loss_bbox_0": 5.0,
        "loss_ce": 1.0,
    }
    apply = st.loss_weight(wd, "loss_giou")
    apply(4.0, dry_run=True)
    assert wd["loss_giou"] == 2.0
    apply(4.0, dry_run=False)
    assert {k: v for k, v in wd.items() if k.startswith("loss_giou")} == dict.fromkeys(
        ["loss_giou", "loss_giou_0", "loss_giou_1", "loss_giou_enc", "loss_giou_dn_0"], 4.0
    )
    assert wd["loss_bbox"] == 5.0 and wd["loss_bbox_0"] == 5.0 and wd["loss_ce"] == 1.0


def test_loss_weight_prefix_does_not_bleed() -> None:
    wd = {"loss": 1.0, "loss_bbox": 5.0, "loss_bbox_0": 5.0}
    st.loss_weight(wd, "loss")(3.0, dry_run=False)
    assert wd == {"loss": 3.0, "loss_bbox": 5.0, "loss_bbox_0": 5.0}


def test_loss_weight_unknown_key() -> None:
    with pytest.raises(ValueError, match="no matching key"):
        st.loss_weight({"loss_ce": 1.0}, "loss_mask")(1.0, dry_run=True)


def test_attribute_and_value_only() -> None:
    target = SimpleNamespace(clip_max_norm=0.1)
    st.attribute(target, "clip_max_norm", st.number("clip_max_norm"))(0.5, dry_run=False)
    assert target.clip_max_norm == 0.5
    with pytest.raises(AttributeError):
        st.attribute(target, "nope", st.number("nope"))
    accept = st.value_only(st.integer("grad_accum_steps", minimum=1, maximum=8))
    accept(4, dry_run=False)
    with pytest.raises(ValueError):
        accept(4.5, dry_run=True)
    with pytest.raises(ValueError):
        accept(9, dry_run=True)
    with pytest.raises(ValueError):
        st.value_only(st.boolean("flag"))(1, dry_run=True)
