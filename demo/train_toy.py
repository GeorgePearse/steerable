"""Toy detector: a small conv net regresses one box per synthetic image with L1 + GIoU.

Runs on CPU in about a minute. Calls ``Steerable.sync(step)`` every step and
``sync(step, at_validation=True)`` at each step-based validation, where it also saves weights.
Connects as steerable_trainer only.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch
from torch import Tensor, nn

from steerable import Policy, Steerable
from steerable import torch as st

log = logging.getLogger("demo.trainer")

POLICIES = [
    Policy("lr", {"type": "number", "min": 1e-6, "max": 1e-1}, mode="auto", max_rel_change=3, max_per_hour=5),
    Policy(
        "loss_giou", {"type": "number", "min": 0, "max": 20}, mode="auto", max_rel_change=3, max_per_hour=5
    ),
    Policy(
        "loss_bbox", {"type": "number", "min": 0, "max": 20}, mode="auto", max_rel_change=3, max_per_hour=5
    ),
    Policy("clip_max_norm", {"type": "number", "min": 0, "max": 100}, mode="auto"),
    Policy("batch_size", {"type": "integer", "min": 4, "max": 256}, boundary="validation", mode="auto"),
    Policy("weight_decay", {"type": "number", "min": 0, "max": 1}, mode="review"),
    Policy(
        "image_size",
        {"type": "integer", "min": 32, "max": 256},
        boundary="frozen",
        note="fixes the input tensor shape the net was built for",
    ),
]


def synth_batch(n: int, size: int, gen: torch.Generator) -> tuple[Tensor, Tensor]:
    images = torch.rand(n, 3, size, size, generator=gen) * 0.2
    boxes = torch.empty(n, 4)
    for i in range(n):
        w, h = torch.rand(2, generator=gen) * 0.5 + 0.15
        cx = torch.rand(1, generator=gen) * (1 - w) + w / 2
        cy = torch.rand(1, generator=gen) * (1 - h) + h / 2
        x0, y0 = int((cx - w / 2) * size), int((cy - h / 2) * size)
        x1, y1 = max(x0 + 1, int((cx + w / 2) * size)), max(y0 + 1, int((cy + h / 2) * size))
        images[i, :, y0:y1, x0:x1] = torch.rand(3, 1, 1, generator=gen) * 0.6 + 0.4
        boxes[i] = torch.tensor([cx.item(), cy.item(), w.item(), h.item()])
    return images, boxes


def cxcywh_to_xyxy(b: Tensor) -> Tensor:
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)


def giou(pred: Tensor, target: Tensor) -> Tensor:
    p, t = cxcywh_to_xyxy(pred), cxcywh_to_xyxy(target)
    lt, rb = torch.max(p[:, :2], t[:, :2]), torch.min(p[:, 2:], t[:, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    area_p = (p[:, 2:] - p[:, :2]).clamp(min=0).prod(-1)
    area_t = (t[:, 2:] - t[:, :2]).clamp(min=0).prod(-1)
    union = area_p + area_t - inter
    iou = inter / union.clamp(min=1e-6)
    elt, erb = torch.min(p[:, :2], t[:, :2]), torch.max(p[:, 2:], t[:, 2:])
    enclose = (erb - elt).clamp(min=0).prod(-1)
    return iou - (enclose - union) / enclose.clamp(min=1e-6)


class ToyDetector(nn.Module):
    def __init__(self, size: int = 64) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Linear(32 * (size // 8) ** 2, 128), nn.ReLU(), nn.Linear(128, 4))

    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(self.head(self.body(x)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True, help="steerable_trainer DSN")
    ap.add_argument("--run", type=int, required=True)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--val-every", type=int, default=100)
    ap.add_argument("--step-sleep", type=float, default=0.05, help="pace the demo so a tuner can react")
    ap.add_argument("--out", default="demo/out")
    ap.add_argument("--threads", type=int, default=1, help="torch CPU threads; 1 is fastest on a shared box")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")

    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    config = {
        "lr": 1e-3,
        "loss_giou": 2.0,
        "loss_bbox": 5.0,
        "clip_max_norm": 0.1,
        "batch_size": 32,
        "image_size": 64,
        "weight_decay": 1e-4,
    }
    model = ToyDetector(config["image_size"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    weight_dict = {"loss_bbox": config["loss_bbox"], "loss_giou": config["loss_giou"]}
    val_images, val_boxes = synth_batch(128, config["image_size"], torch.Generator().manual_seed(1))

    appliers = {
        "lr": st.optimizer_lr(optimizer),
        "weight_decay": st.optimizer_weight_decay(optimizer),
        "loss_giou": st.loss_weight(weight_dict, "loss_giou"),
        "loss_bbox": st.loss_weight(weight_dict, "loss_bbox"),
        "clip_max_norm": st.value_only(st.number("clip_max_norm", minimum=0.0)),
        "batch_size": st.value_only(st.integer("batch_size", minimum=4, maximum=256)),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with Steerable(args.dsn, args.run, appliers, poll_interval_s=0.5) as steer:
        steer.launch(config, POLICIES)
        log.info("run %s launched with config %s", args.run, config)
        for step in range(1, args.steps + 1):
            at_validation = step % args.val_every == 0
            steer.sync(step, at_validation=at_validation)
            batch_size = int(steer.value("batch_size"))
            images, boxes = synth_batch(batch_size, config["image_size"], gen)
            pred = model(images)
            l1 = (pred - boxes).abs().sum(-1).mean()
            g = (1 - giou(pred, boxes)).mean()
            loss = weight_dict["loss_bbox"] * l1 + weight_dict["loss_giou"] * g
            optimizer.zero_grad()
            loss.backward()
            clip = float(steer.value("clip_max_norm"))
            if clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            steer.log_metric(step, "loss", loss.item())
            steer.log_metric(step, "lr", optimizer.param_groups[0]["lr"])
            if step % 50 == 0:
                log.info(
                    "step %d loss %.4f lr %g loss_giou %g batch %d",
                    step,
                    loss.item(),
                    optimizer.param_groups[0]["lr"],
                    weight_dict["loss_giou"],
                    batch_size,
                )
            if at_validation:
                with torch.no_grad():
                    iou = giou(model(val_images), val_boxes).clamp(min=0).mean().item()
                steer.log_metric(step, "val_iou", iou)
                torch.save(model.state_dict(), out / "weights.pt")
                log.info("validation at step %d: mean IoU %.3f (weights saved)", step, iou)
            if args.step_sleep:
                time.sleep(args.step_sleep)
        steer.finish("finished")
    log.info("run %s finished", args.run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
