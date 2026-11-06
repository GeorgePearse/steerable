# steerable

Steer a training run from Postgres: typed, rate-limited, auditable config changes a trainer applies live, with a credential an LLM can hold safely.

An LLM (or a person) proposes a change such as "halve the learning rate". Postgres decides whether the proposal is allowed: the run is running and not frozen, the key is in the run's policy table, the value matches the key's typed schema, the relative change is within bounds, enough steps have passed since the key last changed, and the rate limits hold. The training process polls for approved proposals, dry-runs every applier on every rank, applies the batch at the next step boundary and acknowledges each one. Every proposal and every event is append-only. The database is the policy engine; the LLM's credential can execute one function and read two views, and nothing else.

## 60-second demo

Requires Docker and [uv](https://docs.astral.sh/uv/). CPU only.

```sh
git clone https://github.com/GeorgePearse/steerable && cd steerable
docker compose up -d
make demo     # about 2 minutes: toy detector + a tuner that holds only the agent credential
make attack   # everything that credential might try, refused by Postgres
```

`make demo` creates a run as the admin, starts a toy box-regression trainer (small conv net, L1 + GIoU loss, validation every 100 steps, weights saved) that calls `sync()` every step, and starts a scripted tuner connected as `steerable_agent`. The tuner halves the learning rate after step 200 and doubles the GIoU weight after step 400. Real output:

```
[admin]   created run 3 (demo-20261106-034553)
[trainer] 03:46:02 demo.trainer run 3 launched with config {'lr': 0.001, 'loss_giou': 2.0, 'loss_bbox': 5.0, 'clip_max_norm': 0.1, 'batch_size': 32, 'image_size': 64, 'weight_decay': 0.0001}
[trainer] 03:46:13 demo.trainer validation at step 100: mean IoU 0.734 (weights saved)
[trainer] 03:46:24 demo.trainer step 200 loss 0.7897 lr 0.001 loss_giou 2 batch 32
[trainer] 03:46:24 demo.trainer validation at step 200: mean IoU 0.820 (weights saved)
[tuner]   proposed lr=0.0005: proposal 7 approved
[trainer] 03:46:25 steerable applied proposal 7 lr 0.001 -> 0.0005 at step 206
[trainer] 03:46:29 demo.trainer step 250 loss 0.5118 lr 0.0005 loss_giou 2 batch 32
[trainer] 03:46:46 demo.trainer validation at step 400: mean IoU 0.892 (weights saved)
[tuner]   proposed loss_giou=4.0: proposal 8 approved
[trainer] 03:46:47 steerable applied proposal 8 loss_giou 2 -> 4 at step 407
[trainer] 03:46:51 demo.trainer step 450 loss 0.6506 lr 0.0005 loss_giou 4 batch 32
[trainer] 03:47:32 demo.trainer validation at step 800: mean IoU 0.913 (weights saved)
[trainer] 03:47:32 demo.trainer run 3 finished
[tuner]   run 3 is finished; tuner exiting
[admin]   history:
[admin]   {"proposal_id": 7, "key": "lr", "previous": 0.001, "value": 0.0005, "status": "applied", "status_step": 206, "reason": "loss plateaued over the last 100 steps; halving lr"}
[admin]   {"proposal_id": 8, "key": "loss_giou", "previous": 2.0, "value": 4.0, "status": "applied", "status_step": 407, "reason": "val IoU lags the L1 term; weight GIoU more"}
```

## Three roles

`steerable db init` creates three Postgres roles on stock PostgreSQL 14+ with no extensions. `REVOKE ALL ON SCHEMA steerable FROM PUBLIC` runs first; each role is then granted exactly the surface below.

| Role | Who logs in as it | Can do | Cannot do |
|---|---|---|---|
| `steerable_agent` | an LLM, or `steerable-mcp` acting for one | `EXECUTE steerable.propose(run_id, key, value, reason)`; `SELECT` on the `run_state` and `history` views | read or write any table, acknowledge, decide, change policy, create objects |
| `steerable_trainer` | the training process | `SELECT` on runs, policies, proposals, events, metrics and the `pending` / `current_config` views; `INSERT` metrics; `EXECUTE ack()`, `launch()` (once, on a created run), `finish()` | propose, decide, change policy or runs |
| `steerable_admin` | humans, CI | owns the schema and the `SECURITY DEFINER` functions; writes `runs` and `policies`; `decide()` on review-mode proposals; `UPDATE runs SET agent_enabled = false` freezes a run instantly | `UPDATE`/`DELETE`/`TRUNCATE` proposals or events: a trigger refuses them for every role |

The functions are `SECURITY DEFINER`, owned by `steerable_admin`, with `SET search_path = steerable, pg_temp`, `REVOKE EXECUTE FROM PUBLIC`, an explicit grant to the one role that needs each, every argument validated, and no dynamic SQL. Rejections raise with a stable SQLSTATE in class `SE` and a plain message a caller may hand back to an LLM as data.

Views run with their owner's privileges, so the agent reads `run_state` without any grant on the tables behind it. `run_state` carries numbers, enums, the policy limits and the agent's own past reasons. Policy notes and trainer-side free text stay out of it.

## The policy table

A key with no `policies` row cannot be proposed. Each row says what may change, when, by how much and how often:

| Column | Meaning |
|---|---|
| `schema` | `{"type":"number","min":..,"max":..}`, `integer`, `boolean`, or `{"type":"enum","values":[..]}`. Validated in PL/pgSQL on insert and on every proposal. |
| `boundary` | `step`: applied before the next training step. `validation`: held until the trainer's next validation boundary. `frozen`: cannot be proposed; `note` says why. |
| `mode` | `auto`: approved on insert. `review`: waits for `decide()` by an admin. `frozen`: same as the boundary. |
| `max_rel_change` | largest ratio between the effective value and the proposed one (3 means at most 3x up or down). Sign flips and changes to zero have no finite ratio and are refused when set. |
| `min_steps_between` | steps that must pass after the key was last applied. |
| `max_per_hour` | proposals per hour for this key. `runs.max_proposals_per_hour` caps the whole run as well. |

The effective config is the run's launch config folded with its `applied` events in order. A new proposal for a key supersedes any still-open older proposal for the same key.

## Attack transcript

`make attack` connects as `steerable_agent` against a running run with `lr = 0.001`, `max_rel_change 3`, `max_per_hour 5`, and tries what a hostile or confused agent might try. Real output:

```
connected as steerable_agent

UPDATE steerable.proposals SET value = '0.1'
    -> 42501 permission denied for table proposals
DELETE FROM steerable.events
    -> 42501 permission denied for table events
INSERT INTO steerable.policies (...)
    -> 42501 permission denied for table policies
SELECT * FROM steerable.runs
    -> 42501 permission denied for table runs
SELECT * FROM steerable.proposals
    -> 42501 permission denied for table proposals
UPDATE steerable.runs SET agent_enabled = true
    -> 42501 permission denied for table runs
SELECT steerable.decide(1, 'approved')
    -> 42501 permission denied for function decide
SELECT steerable.ack(1, 1, 'applied')
    -> 42501 permission denied for function ack
CREATE TABLE steerable.notes (t text)
    -> 42501 permission denied for schema steerable
propose unknown key 'optimizer'
    -> SE003 key optimizer has no policy for run 1 and cannot be proposed
propose lr = 0.5 (schema max 0.1)
    -> SE005 value 0.5 does not match the schema {"max": 0.1, "min": 0.000001, "type": "number"} for key lr
propose lr = "0.0005" (string, not number)
    -> SE005 value "0.0005" does not match the schema {"max": 0.1, "min": 0.000001, "type": "number"} for key lr
propose lr = 0.01 (10x jump, max_rel_change 3)
    -> SE006 change of lr from 0.001 to 0.01 is 10x; max_rel_change is 3x
propose frozen key image_size = 128
    -> SE004 key image_size is frozen for run 1: fixes the input tensor shape
propose with empty reason
    -> SE009 reason must be 1 to 1000 characters
propose on a run that does not exist
    -> SE001 run 999999 does not exist

five valid lr proposals, then a sixth inside the hour (max_per_hour 5):
    proposal 1 approved
    proposal 2 approved
    proposal 3 approved
    proposal 4 approved
    proposal 5 approved
propose lr sixth time
    -> SE008 key lr already has 5 proposals in the last hour; max_per_hour is 5

admin freezes the run (UPDATE runs SET agent_enabled = false); the agent tries again:
propose loss_giou = 3 on the frozen run
    -> SE002 run 1 is frozen: proposals are disabled until an admin re-enables them

18/18 attempts refused by the database
```

The threat model behind this is in [docs/threat-model.md](docs/threat-model.md).

## How a change flows

```mermaid
sequenceDiagram
    participant A as LLM via steerable-mcp (steerable_agent)
    participant DB as Postgres (schema steerable)
    participant H as Admin (steerable_admin)
    participant T as Trainer (steerable_trainer)
    A->>DB: propose(run_id, key, value, reason)
    DB->>DB: running and not frozen? policy row? schema? max_rel_change? min_steps_between? rate limits?
    alt policy mode = auto
        DB-->>A: (proposal_id, approved)
    else policy mode = review
        DB-->>A: (proposal_id, pending)
        H->>DB: decide(proposal_id, approved | rejected, note)
    end
    T->>DB: poll pending where status = approved (background thread)
    T->>T: sync(step): dry-run every applier on every rank
    T->>T: apply the survivors in proposal order, hold validation keys
    T->>DB: ack(proposal_id, step, applied | trainer_rejected, note)
    A->>DB: run_state / history show the outcome
```

## Trainer side

```python
from steerable import Steerable, Policy
from steerable import torch as st

appliers = {
    "lr": st.optimizer_lr(optimizer, scheduler=scheduler),
    "weight_decay": st.optimizer_weight_decay(optimizer),
    "loss_giou": st.loss_weight(criterion.weight_dict, "loss_giou"),   # reaches loss_giou_0 .. _enc too
    "ema_decay": st.ema_decay(ema),
    "clip_max_norm": st.value_only(st.number("clip_max_norm")),       # the loop reads steer.value("clip_max_norm")
}
policies = [
    Policy("lr", {"type": "number", "min": 1e-7, "max": 1e-1}, mode="auto", max_rel_change=3, min_steps_between=50),
    Policy("batch_size", {"type": "integer", "min": 1, "max": 512}, boundary="validation", mode="auto"),
    Policy("resolution", {"type": "integer", "min": 64, "max": 2048}, boundary="frozen", note="sizes the head"),
]

with Steerable(TRAINER_DSN, run_id, appliers, poll_interval_s=5.0) as steer:
    steer.launch(config, policies)                       # once; records launch config, moves the run to running
    for step, batch in enumerate(loader, start=1):
        steer.sync(step, at_validation=(step % val_every == 0))
        ...                                              # train; clip with steer.value("clip_max_norm")
        steer.log_metric(step, "loss", loss.item())
    steer.finish()
```

Four public types: `Steerable`, `Applier` (a callable `(value, *, dry_run)`), `Proposal`, `Policy`. A background thread does all database IO, so the loop never blocks. `sync()` dry-runs every applier in the batch first; a proposal whose dry run raises is acknowledged `trainer_rejected` with the exception text and the rest are applied together. Under `torch.distributed`, rank 0 reads the batch and broadcasts it, dry-run failures are gathered from every rank, and all ranks apply the same proposals at the same step. Values are data: no pickle, no eval, no class loaded from the database. Appliers are registered in the process; the database can only pick among allowlisted keys with typed values.

## RF-DETR and D-FINE

Both adapters ship a policy table and bind the upstream objects to keys. The per-knob tables and the reason each frozen key is frozen are in [docs/adapters.md](docs/adapters.md), along with a 30-line recipe for any other DETR-style loop.

RF-DETR (`pip install rfdetr`, 1.11+) trains through PyTorch Lightning and `train()` has no callback parameter, so the adapter attaches a Lightning callback through `build_trainer`:

```python
from rfdetr import RFDETRMedium
from steerable.adapters.rfdetr import SteerableCallback, attach

with attach(SteerableCallback(TRAINER_DSN, run_id)):
    RFDETRMedium().train(dataset_dir="data/", epochs=20)
```

D-FINE (clone of github.com/Peterande/D-FINE) has no callback surface; the adapter wraps the train dataloader so `sync()` runs before every batch and at epoch end, where `fit()` evaluates:

```python
from src.solver import DetSolver                       # inside the D-FINE repo
from steerable.adapters.dfine import steerable_solver

Solver = steerable_solver(DetSolver, dsn=TRAINER_DSN, run_id=run_id)
Solver(cfg).fit()
```

## LLM side

`steerable-mcp` is a stdio MCP server with three tools: `run_state(run_id)`, `propose(run_id, key, value, reason)`, `history(run_id)`. It refuses to start unless its DSN logs in as `steerable_agent`. Database rejections come back as tool results (`accepted: false`, SQLSTATE, message), never as exceptions. For Claude Code:

```sh
uv pip install "steerable[mcp]"
claude mcp add steerable -e STEERABLE_AGENT_DSN=postgresql://steerable_agent:<password>@host:5432/steerable -- steerable-mcp
```

Any MCP client works the same way: command `steerable-mcp`, environment `STEERABLE_AGENT_DSN`.

`steerable tune --run <id>` is a reference autonomous tuner (`steerable[tune]`, Anthropic SDK). It sends only the structured `run_state` row to `claude-opus-5-5` and routes every answer through `propose()`. Without `--live` it prints the state and makes no API call; `--max-usd` stops it before the estimated spend passes a limit.

## Human CLI

```sh
export STEERABLE_SUPERUSER_DSN=postgresql://postgres:postgres@localhost:5440/steerable
export STEERABLE_ADMIN_PASSWORD=... STEERABLE_TRAINER_PASSWORD=... STEERABLE_AGENT_PASSWORD=...
steerable db init                                 # roles, schema, functions, grants (idempotent)
steerable --dsn $ADMIN run create my-run          # prints the run id; the trainer launches it
steerable --dsn $AGENT propose --run 7 lr 0.0005 --reason "loss plateau"
steerable --dsn $ADMIN approve 12 --note "agreed" # review-mode proposals
steerable --dsn $ADMIN freeze --run 7             # agent_enabled = false, instantly
steerable --dsn $ADMIN state --run 7
steerable --dsn $ADMIN history --run 7
```

## Development

```sh
docker compose up -d
make test        # pytest against the compose database (SQL contract, client, appliers, adapters, MCP, tuner, CLI)
make lint typecheck
```

CI runs the same against a `postgres:16` service. Tests that need `rfdetr` or D-FINE installed are marked `integration` and are not part of CI; the adapters are unit-tested with fakes.

## Licence

MIT.
