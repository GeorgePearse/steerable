# Three roles, three passwords. These are demo values for the compose database only.
export STEERABLE_SUPERUSER_DSN ?= postgresql://postgres:postgres@localhost:5440/steerable
export STEERABLE_ADMIN_PASSWORD ?= admin-demo-7f3a
export STEERABLE_TRAINER_PASSWORD ?= trainer-demo-c91e
export STEERABLE_AGENT_PASSWORD ?= agent-demo-2b84
ADMIN_DSN   = postgresql://steerable_admin:$(STEERABLE_ADMIN_PASSWORD)@localhost:5440/steerable
TRAINER_DSN = postgresql://steerable_trainer:$(STEERABLE_TRAINER_PASSWORD)@localhost:5440/steerable
AGENT_DSN   = postgresql://steerable_agent:$(STEERABLE_AGENT_PASSWORD)@localhost:5440/steerable

UV ?= uv

.PHONY: help sync db-up db-init demo attack test lint typecheck fmt clean

help:
	@grep -E '^[a-z-]+:' Makefile | cut -d: -f1 | tr '\n' ' '; echo

sync:
	$(UV) sync --extra dev

db-up:
	docker compose up -d --wait

db-init: db-up
	$(UV) run steerable db init

demo: sync db-init
	$(UV) run python demo/run_demo.py --admin-dsn "$(ADMIN_DSN)" --trainer-dsn "$(TRAINER_DSN)" --agent-dsn "$(AGENT_DSN)"

attack: sync db-init
	$(UV) run python demo/attack.py --admin-dsn "$(ADMIN_DSN)" --trainer-dsn "$(TRAINER_DSN)" --agent-dsn "$(AGENT_DSN)"

test: sync db-up
	STEERABLE_TEST_DSN=$(STEERABLE_SUPERUSER_DSN) $(UV) run pytest -q

lint: sync
	$(UV) run ruff check .
	$(UV) run ruff format --check .

typecheck: sync
	$(UV) run mypy

fmt: sync
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

clean:
	docker compose down -v
	rm -rf .pytest_cache .mypy_cache .ruff_cache demo/out
