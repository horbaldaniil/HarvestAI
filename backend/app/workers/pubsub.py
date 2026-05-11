"""Tiny Redis pub/sub helper for job progress events.

Producers (RQ jobs) publish to `job:{job_id}`. The SSE service subscribes
to the same channel and proxies events to the browser.

Event shape (JSON):
  {"state": "queued"|"running"|"done"|"failed", "progress": 0..1, "data": {...}, "error": "..."}
"""
from __future__ import annotations

import json
from typing import Any, Literal

import redis as sync_redis

JobState = Literal["queued", "running", "done", "failed"]


def channel_for(job_id: str) -> str:
    return f"job:{job_id}"


def publish_progress(
    redis: sync_redis.Redis,
    job_id: str,
    state: JobState,
    progress: float | None = None,
    data: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    payload: dict[str, Any] = {"state": state}
    if progress is not None:
        payload["progress"] = max(0.0, min(1.0, progress))
    if data is not None:
        payload["data"] = data
    if error is not None:
        payload["error"] = error
    redis.publish(channel_for(job_id), json.dumps(payload, ensure_ascii=False))
