"""`make demo`: create a run as admin, train as trainer, tune as agent; print both logs; show history."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

HERE = Path(__file__).parent


def pump(prefix: str, proc: subprocess.Popen[str]) -> None:
    assert proc.stdout is not None
    for line in proc.stdout:
        print(f"{prefix} {line.rstrip()}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--admin-dsn", required=True)
    ap.add_argument("--trainer-dsn", required=True)
    ap.add_argument("--agent-dsn", required=True)
    ap.add_argument("--steps", type=int, default=800)
    args = ap.parse_args()

    name = f"demo-{time.strftime('%Y%m%d-%H%M%S')}"
    with psycopg.connect(args.admin_dsn, row_factory=dict_row, autocommit=True) as admin:
        row = admin.execute("INSERT INTO steerable.runs (name) VALUES (%s) RETURNING id", (name,)).fetchone()
        assert row is not None
        run_id = int(row["id"])
    print(f"[admin]   created run {run_id} ({name})", flush=True)

    trainer = subprocess.Popen(
        [
            sys.executable,
            str(HERE / "train_toy.py"),
            "--dsn",
            args.trainer_dsn,
            "--run",
            str(run_id),
            "--steps",
            str(args.steps),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    tuner = subprocess.Popen(
        [sys.executable, str(HERE / "tuner.py"), "--dsn", args.agent_dsn, "--run", str(run_id)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    threads = [
        threading.Thread(target=pump, args=("[trainer]", trainer), daemon=True),
        threading.Thread(target=pump, args=("[tuner]  ", tuner), daemon=True),
    ]
    for t in threads:
        t.start()
    trainer_rc = trainer.wait()
    try:
        tuner.wait(timeout=10)
    except subprocess.TimeoutExpired:
        tuner.kill()
    for t in threads:
        t.join(timeout=5)

    with psycopg.connect(args.admin_dsn, row_factory=dict_row, autocommit=True) as admin:
        rows = admin.execute(
            "SELECT proposal_id, key, previous, value, status, status_step, reason FROM steerable.history "
            "WHERE run_id = %s ORDER BY proposal_id",
            (run_id,),
        ).fetchall()
    print("[admin]   history:", flush=True)
    for r in rows:
        print("[admin]   " + json.dumps(r, default=str), flush=True)
    return trainer_rc


if __name__ == "__main__":
    raise SystemExit(main())
