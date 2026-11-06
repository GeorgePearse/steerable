# Threat model

The design goal: an LLM may hold the `steerable_agent` credential, and whatever it does with it, the worst outcome is a bounded, typed, rate-limited, logged change to an allowlisted training knob, or no change at all. No Python sits between the credential and the policy. If the MCP server or the agent is prompt-injected or buggy, the blast radius is set by Postgres grants and the policy table, not by application code.

## Components and trust

| Component | Credential | Trusted to |
|---|---|---|
| LLM, `steerable-mcp`, `steerable tune`, scripted tuners | `steerable_agent` | propose values for allowlisted keys and read `run_state` / `history`. Nothing else. |
| Training process (`Steerable`, adapters) | `steerable_trainer` | apply approved proposals faithfully, report metrics and acknowledgements, launch and finish its own run. |
| Humans, CI, `steerable db init` | `steerable_admin` or a superuser | write runs and policies, decide review-mode proposals, freeze runs. |
| Postgres | n/a | enforce grants, triggers, constraints and the `propose()` checks. The policy engine. |

## Threats and the mechanism that bounds each

| Threat | What it could try | What bounds it |
|---|---|---|
| Wrong LLM: a model that is confidently mistaken, or that optimises the wrong thing | propose a 100x learning rate, a negative loss weight, a resolution the model was not built for, dozens of changes a minute | `policies.schema` (typed, with min/max/enum), `max_rel_change`, `min_steps_between`, `max_per_hour`, `runs.max_proposals_per_hour`, `boundary = frozen` for keys that must not move. `mode = review` puts a human in front of risky keys. Every change is a proposal with a reason, visible in `history`, reversible by proposing the previous value, and the admin can `freeze` the run in one statement. |
| Prompt-injected LLM: instructions smuggled through metric names, reasons, or any text it reads | write to other tables, approve its own proposals, disable the freeze, exfiltrate the policy notes or runs table | The credential has `EXECUTE` on `propose()` and `SELECT` on two views only. There is no grant path to `UPDATE`, `DELETE`, `INSERT`, `decide()`, `ack()`, `launch()`, or any table. `run_state` carries numbers, enums, policy limits and the agent's own past reasons; policy notes and the `actor` column are not in it. Rejection messages are returned as data. The only free text the agent can read that it did not write is the trainer's `trainer_rejected` note (see "compromised trainer"). |
| Leaked agent credential: the DSN ends up in a log, a prompt, or a repository | anything the agent could do, from anywhere that can reach the database | Same bound as above, plus `CONNECTION LIMIT 8` on the role, the run-level rate limit, and `runs.agent_enabled = false` as the kill switch. Rotate with `ALTER ROLE steerable_agent PASSWORD ...`; nothing else changes. Proposals record `actor` (the login role) and `actor_kind`, so the audit trail survives. |
| SQL injection through the MCP server or CLI | a key such as `lr'; DROP TABLE ...` or a reason containing SQL | Every query is parameterised (psycopg sends values out of band). Keys must match `^[a-z][a-z0-9_.]{0,63}$` and be a policy row; run names match `^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$`. The `SECURITY DEFINER` functions build no dynamic SQL and run with `search_path = steerable, pg_temp`, so a schema or function planted elsewhere cannot be picked up. Even a successful injection in a client runs as `steerable_agent` and hits the same grants. |
| Compromised trainer process | forge approvals, rewrite history, alter policy, propose on its own behalf | `steerable_trainer` cannot `propose()` or `decide()`, cannot write runs or policies, and can only append `applied` / `trainer_rejected` events to proposals that already carry an `approved` event. Proposals and events are append-only for every role including the owner. Its one free-text channel is the `trainer_rejected` note (at most 2000 characters), which an agent may read in `history`; treat it as data. `launch()` is accepted once, on a `created` run; policy rows an admin seeded first are kept, so an admin can pin the policy before the trainer starts. Per-run isolation between several trainers sharing the role is not provided; use one database per trust domain if that matters. |
| Code execution through config values | a value such as `{"__class__": ...}`, a pickle, an import path, a lambda | Values are JSON validated against a tiny schema (`number`, `integer`, `boolean`, `enum`) in PL/pgSQL; objects and arrays never pass. The trainer never imports, evals or unpickles anything from the database. Appliers are registered in the training process, so a value can only reach the mutation that the process chose for that key, with the applier's own range check on top. |
| Torn application mid-step | an LR change lands on half the ranks, or a loss weight changes between forward and backward | `sync()` runs only at the step boundary the loop chooses. Rank 0 reads the batch and broadcasts it; every rank dry-runs every applier; the failing set is gathered and unioned; all ranks then apply the same survivors in the same order. `validation` keys are held until `sync(at_validation=True)`. A proposal is acknowledged `applied` only after every rank has applied it. If the process dies between apply and ack, the proposal stays `approved` and is re-applied on restart from the same effective value, which is idempotent for every applier shipped here. |
| Malicious or careless admin | set `mode = auto` on a dangerous key, drop the trigger, truncate history | Out of scope: the admin owns the schema. Keep the admin DSN away from agents and automation that does not need it. The append-only triggers make accidental history edits fail loudly; dropping them is a deliberate, visible act. |

## Things this design does not do

- It does not authenticate the LLM. Anyone with the agent DSN is the agent. Treat it like any other service credential.
- It does not stop a bad but policy-conforming change. That is what `max_rel_change`, `min_steps_between`, `review` mode and the metrics in `run_state` are for; tune them per run.
- It does not isolate runs from each other for the trainer role. Row-level security keyed on a per-run trainer role is a straightforward extension if several untrusted trainers share one database.
- It does not hide table names: `pg_catalog` is readable by every role, so the agent can see that `steerable.policies` exists. It cannot read a row of it.
- It does not run the LLM. `steerable tune` is a reference loop with a spend guard; `steerable-mcp` lets any MCP client be the loop.

## Verifying the bounds

`make attack` exercises eighteen attempts against a live database and expects all of them to fail with Postgres's own error. `tests/test_sql_contract.py` covers the same ground per SQLSTATE, plus grant enumeration per role (`test_each_role_sees_only_its_own_surface`), trigger immutability for the owner, supersession, and the fold of `applied` events into the effective config.
