"""Job status: SSE stream + one-shot status fetch.

The SSE stream is the primary path; the one-shot GET is a fallback for
environments where SSE doesn't make it through (corporate proxies, etc.).
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from rq.exceptions import NoSuchJobError
from rq.job import Job

from app.deps import CurrentUser
from app.redis_clients import get_async_redis, get_sync_redis
from app.schemas.observation import JobStatusResponse
from app.services.sse_service import stream_job_events

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.get("/{job_id}/stream")
async def stream(job_id: str, current_user: CurrentUser) -> StreamingResponse:
    """Server-Sent Events stream for a specific job's progress.

    The browser opens this with new EventSource("/api/jobs/{id}/stream") and
    receives `status` events as the job progresses.
    """
    return StreamingResponse(
        stream_job_events(get_async_redis(), job_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",  # disable nginx buffering when behind it
        },
    )


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_status(job_id: str, current_user: CurrentUser) -> JobStatusResponse:
    """Fallback one-shot status — for clients that can't use SSE."""
    try:
        job = Job.fetch(job_id, connection=get_sync_redis())
    except NoSuchJobError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found") from exc
    state = job.get_status() or "unknown"
    # Map RQ states to our literal type.
    mapping = {
        "queued": "queued",
        "started": "running",
        "deferred": "queued",
        "scheduled": "queued",
        "finished": "done",
        "failed": "failed",
    }
    return JobStatusResponse(
        job_id=job_id,
        state=mapping.get(state, "unknown"),
        result=job.result if state == "finished" else None,
    )
