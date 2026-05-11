"""Entrypoint for the RQ worker process.

Why SimpleWorker (not the default `Worker`):
- The default `rq.Worker` uses `os.fork()` to spawn a child for each job —
  which doesn't exist on Windows. SimpleWorker runs the job inline in the
  same process, which works cross-platform and is fine for our scale
  (jobs are short, < 30s each, and run sequentially).
- On Linux/Docker (Railway), SimpleWorker still works correctly — we lose
  per-job process isolation but gain platform portability.

Usage (from the backend/ directory):

    Windows:
        uv run python -m app.workers.rq_worker

    Linux/macOS:
        uv run python -m app.workers.rq_worker
        # or the standard rq CLI:
        uv run rq worker --worker-class rq.SimpleWorker default high low

Or via Makefile target `make worker`. In Docker / Railway the same image
is launched with command `python -m app.workers.rq_worker`.
"""
from __future__ import annotations

import logging
import sys

from rq import SimpleWorker

from app.config import settings
from app.redis_clients import get_sync_redis


def main() -> int:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    settings.ensure_directories()
    queues = sys.argv[1:] or ["default"]
    worker = SimpleWorker(queues, connection=get_sync_redis())
    worker.work(with_scheduler=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
