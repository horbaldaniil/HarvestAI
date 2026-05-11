"""Server-Sent Events bridge: Redis pub/sub channel → HTTP text/event-stream.

Why SSE (not WebSockets):
- Pure HTTP — no special proxy/load-balancer config required.
- One-way (server → client) is enough for job progress notifications.
- Built into the browser via EventSource — zero client-side libraries.

Why pub/sub (not polling Redis with GET):
- RQ workers publish events with sub-second granularity; pub/sub propagates
  them with near-zero latency. Polling would either be wasteful or laggy.
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator

from redis.asyncio import Redis

from app.workers.pubsub import channel_for

log = logging.getLogger(__name__)

HEARTBEAT_INTERVAL_S = 15
JOB_MAX_DURATION_S = 600  # safety cap so a stuck channel doesn't hold a connection forever
TERMINAL_STATES = frozenset({"done", "failed"})


async def stream_job_events(redis: Redis, job_id: str) -> AsyncIterator[str]:
    """Yield SSE-framed strings for a given job id.

    Each yielded string is a complete SSE event. The stream ends after a
    terminal state event or when JOB_MAX_DURATION_S is exceeded.
    """
    pubsub = redis.pubsub()
    channel = channel_for(job_id)
    await pubsub.subscribe(channel)
    log.info("SSE subscribed to %s", channel)

    started = asyncio.get_event_loop().time()
    try:
        # Initial "connected" event so the client knows the stream is alive.
        yield _format("status", {"state": "connected"})

        while True:
            elapsed = asyncio.get_event_loop().time() - started
            if elapsed > JOB_MAX_DURATION_S:
                yield _format("status", {"state": "failed", "error": "timeout"})
                return

            try:
                msg = await asyncio.wait_for(
                    pubsub.get_message(ignore_subscribe_messages=True),
                    timeout=HEARTBEAT_INTERVAL_S,
                )
            except asyncio.TimeoutError:
                # Heartbeat keeps proxies from closing idle connections.
                yield ": keepalive\n\n"
                continue

            if msg is None:
                continue

            try:
                data = json.loads(msg["data"])
            except (ValueError, TypeError, KeyError) as exc:
                log.warning("Invalid SSE payload on %s: %s", channel, exc)
                continue

            yield _format("status", data)

            if data.get("state") in TERMINAL_STATES:
                return
    finally:
        await pubsub.unsubscribe(channel)
        await pubsub.aclose()
        log.info("SSE unsubscribed from %s", channel)


def _format(event: str, data: dict) -> str:
    """Format one SSE message. Each event must end with a blank line."""
    body = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {body}\n\n"
