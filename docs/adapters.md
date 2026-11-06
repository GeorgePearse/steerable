# Adapters

An adapter does three things: it binds a loop's live objects (optimizer, criterion, EMA, trainer flags) to steerable keys through appliers, it ships a default policy table saying what may change and when, and it finds the least invasive point in the upstream loop to call `sync()`. Nothing in `steerable.adapters` imports its upstream package at module load, so the tables below can be read, tested and seeded by an admin without the package installed.

Boundaries:

- `step`: applied before the next optimizer step, on every rank, after a dry run on every rank.
- `validation`: held by the trainer until its next validation boundary (`sync(step, at_validation=True)`).
- `frozen`: cannot be proposed. The policy row's `note` is returned in the rejection, so the agent learns why.

## RF-DETR (`steerable.adapters.rfdetr`)

Target: `pip install rfdetr` 1.11+, which trains through PyTorch Lightning. `RFDETR.train()` builds a `Trainer` in `rfdetr.training.trainer.build_trainer` and calls `fit()` immediately, with no callback parameter, so the adapter is a Lightning `Callback` attached through `build_trainer` for the duration of a `with` block. RF-DETR's own callbacks (EMA, drop-path, checkpoint, COCO eval, early stopping) stay in place.

```python
from rfdetr import RFDETRMedium
from steerable.adapters.rfdetr import SteerableCallback, attach

with attach(SteerableCallback(TRAINER_DSN, run_id, poll_interval_s=5.0)):
    RFDETRMedium().train(dataset_dir="data/", epochs=20, batch_size=8)
```

Hook points: `on_fit_start` binds appliers and calls `launch()` with the values read from `TrainConfig` and `ModelConfig`; `on_train_batch_start` calls `sync(trainer.global_step)`; `on_validation_start` calls `sync(step, at_validation=True)` (skipped during the sanity check); `on_train_batch_end` and `on_validation_end` log `train/loss` and the `val/*` metrics Lightning exposes in `callback_metrics`; `on_fit_end` and `on_exception` finish the run.

| Key | Schema | Boundary | Mode | Limits | Applied to |
|---|---|---|---|---|---|
| `lr` | number 1e-8..0.1 | step | auto | 3x, 50 steps apart, 6/h | every param group; `scheduler.base_lrs` rescaled so the LambdaLR/cosine schedule continues from the new rate |
| `weight_decay` | number 0..1 | step | review | 10x | every param group |
| `cls_loss_coef` | number 0..50 | step | auto | 3x, 4/h | `criterion.weight_dict["loss_ce"]` and its `_0..n`, `_enc` copies |
| `bbox_loss_coef` | number 0..50 | step | auto | 3x, 4/h | `loss_bbox` and copies |
| `giou_loss_coef` | number 0..50 | step | auto | 3x, 4/h | `loss_giou` and copies |
| `ema_decay` | number 0.9..0.99999 | step | review | | the `RFDETREMACallback` decay; rejected when the run has no EMA |
| `clip_max_norm` | number 0..100 | step | auto | 4/h | `trainer.gradient_clip_val`, read by Lightning at each optimizer step |
| `resolution` | integer | frozen | | | sizes the positional encoding and every multi-scale target; the model is built for it |
| `batch_size` | integer | frozen | | | Lightning owns the dataloader for the whole fit; a rebuild needs a new run |
| `grad_accum_steps` | integer | frozen | | | Lightning's accumulation window is fixed at Trainer construction |
| `epochs` | integer | frozen | | | the cosine schedule, warmup and checkpoint cadence are sized from it at launch |
| `use_ema` | boolean | frozen | | | EMA weights feed evaluation, best-checkpoint selection and early stopping throughout |
| `amp_dtype` | enum bf16/fp16/fp32 | frozen | | | precision is fixed in the Trainer and the optimizer state is laid out for it |
| `lr_scheduler` | enum step/cosine | frozen | | | the scheduler object is constructed once in `configure_optimizers` |
| `multi_scale` | enum off/per_batch | frozen | | | changes the augmentation distribution the loss landscape was tuned for |

Not in the table on purpose: `num_select`, `drop_path`, `early_stopping_*`. They are reachable (`pl_module.postprocess`, the drop-path callback, the early-stopping callback) and a fork of `bind_appliers` can add them; they were left out to keep the first table to knobs whose effect on a running fit is unambiguous.

## D-FINE (`steerable.adapters.dfine`)

Target: a clone of github.com/Peterande/D-FINE (also RT-DETR and DEIM, which share the solver layout). `DetSolver.fit()` calls `train_one_epoch(model, criterion, self.train_dataloader, self.optimizer, ...)` once per epoch and `evaluate()` after each one. There is no callback surface and `train_one_epoch` is a plain loop over the dataloader, so the dataloader is the hook: `SteerableLoader` wraps it, calls `sync(step)` before yielding each batch and `sync(step, at_validation=True)` when the epoch's iteration ends, which is exactly where `fit()` evaluates. `set_epoch`, `collate_fn`, `sampler`, `batch_size` and `len()` are forwarded, so `fit()` sees no difference.

```python
from src.solver import DetSolver                       # inside the D-FINE repo
from steerable.adapters.dfine import steerable_solver

Solver = steerable_solver(DetSolver, dsn=TRAINER_DSN, run_id=run_id)
Solver(cfg).fit()
```

`steerable_solver` returns a `DetSolver` subclass whose `train()` runs the upstream setup, binds appliers, launches the run and installs the wrapper; `fit()` finishes the run as `finished` or `aborted`. Steps are `epoch * len(loader) + i`, the same count D-FINE logs.

| Key | Schema | Boundary | Mode | Limits | Applied to |
|---|---|---|---|---|---|
| `lr` | number 1e-8..0.1 | step | auto | 3x, 50 steps apart, 6/h | every param group; `lr_scheduler.base_lrs` rescaled so the epoch-based MultiStepLR continues from the new rate |
| `weight_decay` | number 0..1 | step | review | 10x | every param group |
| `loss_vfl`, `loss_bbox`, `loss_giou`, `loss_fgl`, `loss_ddf` | number 0..50 | step | auto | 3x, 4/h | `criterion.weight_dict[key]`; only keys present in the criterion are bound |
| `ema_decay` | number 0.9..0.99999 | step | review | | `ModelEMA.decay` and `decay_fn` together: D-FINE captures the decay inside the lambda, so setting the attribute alone would do nothing |
| `clip_max_norm` | number 0..100 | validation | auto | 4/h | `cfg.clip_max_norm`, which `fit()` passes to `train_one_epoch` once per epoch, so a change takes effect at the next epoch |
| `batch_size` | integer | frozen | | | the DataLoader and its DistributedSampler are built once in `train()` |
| `epochs` | integer | frozen | | | MultiStepLR milestones, `stop_epoch` and the checkpoint cadence are sized from it |
| `stop_epoch` | integer | frozen | | | switches the multi-scale augmentation off and reloads `best_stg1`; a schedule, not a knob |
| `use_amp` | boolean | frozen | | | the GradScaler and autocast context are created at setup |
| `use_ema` | boolean | frozen | | | evaluation and checkpoints read the EMA module throughout the run |
| `eval_spatial_size` | integer | frozen | | | the evaluator and the exported model are fixed to it |
| `num_classes` | integer | frozen | | | sizes the classification head |

## Writing an adapter for another DETR-style loop

RT-DETR, DEIM and LW-DETR all have an optimizer with param groups, a criterion with a `weight_dict`, an optional EMA object, and a loop you can reach either through a callback or through the dataloader. The whole adapter is the mapping from those objects to keys plus a policy table:

```python
from steerable import Policy, Steerable
from steerable import torch as st

POLICIES = [
    Policy("lr", {"type": "number", "min": 1e-8, "max": 0.1}, mode="auto", max_rel_change=3, min_steps_between=50),
    Policy("loss_giou", {"type": "number", "min": 0, "max": 50}, mode="auto", max_rel_change=3),
    Policy("ema_decay", {"type": "number", "min": 0.9, "max": 0.99999}, mode="review"),
    Policy("clip_max_norm", {"type": "number", "min": 0, "max": 100}, mode="auto"),
    Policy("batch_size", {"type": "integer", "min": 1, "max": 512}, boundary="frozen", note="loader built once"),
]

def bind(optimizer, scheduler, criterion, ema):
    return {
        "lr": st.optimizer_lr(optimizer, scheduler=scheduler),
        "loss_giou": st.loss_weight(criterion.weight_dict, "loss_giou"),
        "ema_decay": st.ema_decay(ema),
        "clip_max_norm": st.value_only(st.number("clip_max_norm")),
    }

def train(dsn, run_id, model, optimizer, scheduler, criterion, ema, loader, val_every):
    config = {"lr": optimizer.param_groups[0]["lr"], "loss_giou": criterion.weight_dict["loss_giou"],
              "ema_decay": ema.decay, "clip_max_norm": 0.1, "batch_size": loader.batch_size}
    with Steerable(dsn, run_id, bind(optimizer, scheduler, criterion, ema)) as steer:
        steer.launch(config, POLICIES)
        for step, (samples, targets) in enumerate(loader, start=1):
            steer.sync(step, at_validation=(step % val_every == 0))
            loss = sum(criterion(model(samples), targets).values())
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), steer.value("clip_max_norm"))
            optimizer.step(); ema.update(model)
            steer.log_metric(step, "loss", loss.item())
        steer.finish()
```

Rules of thumb for the policy table:

- A key is `step` only if the object it mutates is read fresh on every step and the mutation is idempotent. Learning rate, weight decay, loss weights, EMA decay and gradient clipping qualify in every loop above.
- A key is `validation` if the loop reads it once per epoch or per validation cycle, or if applying it needs a rebuild the loop already does at that boundary (loader fields, augmentation).
- A key is `frozen` if changing it mid-run invalidates something already materialised: the model's shape, the optimizer's state layout, the schedule's length, the evaluation protocol, the dataset fingerprint. Write the reason in `note`; it is the message the agent gets back.
- Use `review` for anything you would not let a colleague change without telling you, and `max_rel_change` on every continuous knob.
- Give every applier a dry run that checks the value against the live object (group index exists, EMA exists, key exists in `weight_dict`). The exception text is what the agent reads in `history`.
