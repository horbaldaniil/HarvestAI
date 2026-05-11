"""Thin wrappers over `rq.Queue.enqueue` — hides RQ specifics from callers.

The post-field-create pipeline is a chain of four jobs that run in series:
  fetch_sentinel → fetch_weather → predict_yield → check_anomalies

We use RQ's `depends_on` so each step waits for the previous to succeed.
The frontend tracks the LAST job in the chain via SSE — when that one
reports "done", the user knows all four are done.
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


def enqueue_fetch_weather(field_id: int, *, depends_on=None) -> JobHandle:
    job = _queue("default").enqueue(
        "app.workers.jobs.fetch_weather.fetch_field_weather",
        field_id,
        job_timeout=120,
        result_ttl=3600,
        depends_on=depends_on,
    )
    return JobHandle(job_id=job.id, queue=job.origin)


def enqueue_predict_yield(field_id: int, *, depends_on=None) -> JobHandle:
    job = _queue("default").enqueue(
        "app.workers.jobs.predict_yield.predict_field_yield",
        field_id,
        job_timeout=60,
        result_ttl=3600,
        depends_on=depends_on,
    )
    return JobHandle(job_id=job.id, queue=job.origin)


def enqueue_check_anomalies(field_id: int, *, depends_on=None) -> JobHandle:
    job = _queue("default").enqueue(
        "app.workers.jobs.check_anomalies.check_field_anomalies",
        field_id,
        job_timeout=60,
        result_ttl=3600,
        depends_on=depends_on,
    )
    return JobHandle(job_id=job.id, queue=job.origin)


def enqueue_full_pipeline(field_id: int, years_back: int = 2) -> JobHandle:
    """Kick off the full chain after field create / geometry update.

    Returns the handle of the FINAL job. Front-end subscribes to its SSE
    stream; when that says "done", everything is ready.
    """
    sentinel = enqueue_fetch_observations(field_id, years_back)
    # Look up Job objects for depends_on (RQ accepts Job, id, or list).
    weather = enqueue_fetch_weather(field_id, depends_on=sentinel.job_id)
    predict = enqueue_predict_yield(field_id, depends_on=weather.job_id)
    anomalies = enqueue_check_anomalies(field_id, depends_on=predict.job_id)
    return anomalies
