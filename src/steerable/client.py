from __future__ import annotations

import json
import logging
import queue
import threading
import time
from collections.abc import Iterable, Mapping
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from steerable import _dist
from steerable.types import Applier, Policy, Proposal

log = logging.getLogger("steerable")

_PENDING_SQL = """
SELECT proposal_id, run_id, key, value, previous, actor_kind, reason, status, boundary, created_at
FROM steerable.pending
WHERE run_id = %s AND status = 'approved'
ORDER BY proposal_id
"""
_CONFIG_SQL = "SELECT key, value FROM steerable.current_config WHERE run_id = %s"
_METRIC_SQL = "INSERT INTO steerable.metrics (run_id, step, name, value) VALUES (%s, %s, %s, %s)"
_ACK_SQL = "SELECT steerable.ack(%s, %s, %s, %s)"
_LAUNCH_SQL = "SELECT steerable.launch(%s, %s, %s)"
_FINISH_SQL = "SELECT steerable.finish(%s, %s)"


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}".replace("e-0", "e-").replace("e+0", "e+")
    return json.dumps(value)


def _row_to_proposal(row: Mapping[str, Any]) -> Proposal:
    return Proposal(
        id=row["proposal_id"],
        run_id=row["run_id"],
        key=row["key"],
        value=row["value"],
        previous=row["previous"],
        actor_kind=row["actor_kind"],
        reason=row["reason"],
        status=row["status"],
        boundary=row["boundary"],
        created_at=row["created_at"],
    )


class Steerable:
    """Trainer-side client. Connects as ``steerable_trainer``.

    A background thread fetches approved proposals and flushes metrics so the training loop
    never waits on the database. ``sync(step)`` at a step boundary applies the batch that has
    arrived: every applier is dry-run on every rank first, then the surviving proposals are
    applied in order and acknowledged. Keys whose policy boundary is ``validation`` are held
    until ``sync(step, at_validation=True)``. Under torch.distributed, rank 0 reads the batch
    and broadcasts it, so every rank applies the same proposals at the same step.
    """

    def __init__(
        self,
        dsn: str,
        run_id: int,
        appliers: Mapping[str, Applier],
        poll_interval_s: float = 5.0,
    ) -> None:
        self.dsn = dsn
        self.run_id = int(run_id)
        self.appliers = dict(appliers)
        self.poll_interval_s = float(poll_interval_s)
        self._comm = _dist.detect()
        self._config: dict[str, Any] = {}
        self._queue: queue.Queue[Proposal] = queue.Queue()
        self._seen: set[int] = set()
        self._held: dict[str, Proposal] = {}
        self._metrics: list[tuple[int, int, str, float]] = []
        self._metrics_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._main: psycopg.Connection[dict[str, Any]] | None = None
        if self._comm.is_leader:
            self._load_config()
            self._thread = threading.Thread(target=self._poll_loop, name="steerable-poll", daemon=True)
            self._thread.start()
        self._config = self._comm.broadcast(self._config)

    # ----- lifecycle -----

    def __enter__(self) -> Steerable:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.poll_interval_s + 1.0))
            self._thread = None
        if self._comm.is_leader:
            self._flush_metrics(self._conn())
        if self._main is not None:
            self._main.close()
            self._main = None

    def launch(self, config: Mapping[str, Any], policies: Iterable[Policy] = ()) -> None:
        """Record the launch config and default policies; moves the run to ``running``. Once per run."""
        if not self._comm.is_leader:
            return
        payload = [p.to_json() for p in policies]
        self._conn().execute(_LAUNCH_SQL, (self.run_id, Jsonb(dict(config)), Jsonb(payload)))
        self._load_config()

    def finish(self, status: str = "finished") -> None:
        if not self._comm.is_leader:
            return
        self._flush_metrics(self._conn())
        self._conn().execute(_FINISH_SQL, (self.run_id, status))

    # ----- the training loop's surface -----

    def value(self, key: str, default: Any = None) -> Any:
        """Current effective value of ``key`` (launch config folded with applied proposals)."""
        return self._config.get(key, default)

    def log_metric(self, step: int, name: str, value: float) -> None:
        if not self._comm.is_leader:
            return
        with self._metrics_lock:
            self._metrics.append((self.run_id, int(step), name, float(value)))

    def sync(self, step: int, *, at_validation: bool = False) -> list[Proposal]:
        """Apply the proposals that have arrived. Returns the proposals applied at this step."""
        batch = self._collect(at_validation) if self._comm.is_leader else []
        batch = self._comm.broadcast(batch)
        if not batch:
            return []

        failures: dict[int, str] = {}
        for p in batch:
            applier = self.appliers.get(p.key)
            if applier is None:
                failures[p.id] = f"no applier registered for key {p.key}"
                continue
            try:
                applier(p.value, dry_run=True)
            except Exception as exc:
                failures[p.id] = f"{type(exc).__name__}: {exc}"[:2000]
        for other in self._comm.all_gather(failures):
            failures.update(other)

        applied: list[Proposal] = []
        for p in batch:
            if p.id in failures:
                continue
            self.appliers[p.key](p.value, dry_run=False)
            self._config[p.key] = p.value
            applied.append(p)

        if self._comm.is_leader:
            for p in batch:
                if p.id in failures:
                    self._ack(p, step, "trainer_rejected", failures[p.id])
                    log.warning("rejected proposal %s %s: %s", p.id, p.key, failures[p.id])
                else:
                    self._ack(p, step, "applied", None)
                    log.info(
                        "applied proposal %s %s %s -> %s at step %s",
                        p.id,
                        p.key,
                        _fmt(p.previous),
                        _fmt(p.value),
                        step,
                    )
        return applied

    # ----- internals -----

    def _conn(self) -> psycopg.Connection[dict[str, Any]]:
        if self._main is None or self._main.closed:
            self._main = psycopg.connect(self.dsn, row_factory=dict_row, autocommit=True)
        return self._main

    def _load_config(self) -> None:
        rows = self._conn().execute(_CONFIG_SQL, (self.run_id,)).fetchall()
        self._config = {r["key"]: r["value"] for r in rows}

    def _collect(self, at_validation: bool) -> list[Proposal]:
        fresh: list[Proposal] = []
        while True:
            try:
                fresh.append(self._queue.get_nowait())
            except queue.Empty:
                break
        for p in fresh:
            if p.boundary == "validation" and not at_validation:
                self._held[p.key] = p
        ready = [p for p in fresh if not (p.boundary == "validation" and not at_validation)]
        if at_validation:
            ready.extend(self._held.values())
            self._held.clear()
        latest: dict[str, Proposal] = {}
        for p in sorted(ready, key=lambda p: p.id):
            latest[p.key] = p
        return sorted(latest.values(), key=lambda p: p.id)

    def _ack(self, p: Proposal, step: int, status: str, note: str | None) -> None:
        try:
            self._conn().execute(_ACK_SQL, (p.id, int(step), status, note))
        except psycopg.Error as exc:
            log.error("ack of proposal %s failed: %s", p.id, exc)

    def _flush_metrics(self, conn: psycopg.Connection[dict[str, Any]]) -> None:
        with self._metrics_lock:
            rows, self._metrics = self._metrics, []
        if not rows:
            return
        try:
            with conn.cursor() as cur:
                cur.executemany(_METRIC_SQL, rows)
        except psycopg.Error as exc:
            log.error("metric flush failed (%d rows dropped): %s", len(rows), exc)

    def _poll_loop(self) -> None:
        conn: psycopg.Connection[dict[str, Any]] | None = None
        while not self._stop.is_set():
            try:
                if conn is None or conn.closed:
                    conn = psycopg.connect(self.dsn, row_factory=dict_row, autocommit=True)
                rows = conn.execute(_PENDING_SQL, (self.run_id,)).fetchall()
                for row in rows:
                    if row["proposal_id"] not in self._seen:
                        self._seen.add(row["proposal_id"])
                        self._queue.put(_row_to_proposal(row))
                self._flush_metrics(conn)
            except psycopg.Error as exc:
                log.error("poll failed, retrying: %s", exc)
                if conn is not None:
                    conn.close()
                    conn = None
            self._stop.wait(self.poll_interval_s)
        if conn is not None:
            conn.close()

    def wait_for_proposals(self, timeout_s: float) -> bool:
        """Testing aid: block until the poller has queued at least one proposal or the timeout passes."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if not self._queue.empty():
                return True
            time.sleep(0.05)
        return not self._queue.empty()
