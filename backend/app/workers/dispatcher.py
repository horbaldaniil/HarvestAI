"""Thin wrappers over `rq.Queue.enqueue` — hides RQ specifics from callers.

If we ever swap RQ for Kafka/Celery (Week-6/master's thesis extension),
only this module changes.
"""
from __future__ import annotations

from dataclasses import dataclass

from rq import Queue

from app.redis_clients import get_sync_redis


@dataclass(frozen=True, slots=True)
class JobHandle:
    job_id: str
    queue: str


def _queue(name: str = "default") -> Queue:
    return Queue(name, connection=get_sync_redis())


def enqueue_fetch_observations(field_id: int, years_back: int = 2) -> JobHandle:
    job = _queue("default").enqueue(
        "app.workers.jobs.fetch_sentinel.fetch_field_observations",
        field_id,
        years_back,
        job_timeout=300,
        result_ttl=3600,
    )
    return JobHandle(job_id=job.id, queue=job.origin)


def enqueue_fetch_heatmap(field_id: int, target_date: str, index: str) -> JobHandle:
    job = _queue("default").enqueue(
        "app.workers.jobs.fetch_heatmap.fetch_field_heatmap",
        field_id,
        target_date,
        index,
        job_timeout=120,
        result_ttl=3600,
    )
    return JobHandle(job_id=job.id, queue=job.origin)
