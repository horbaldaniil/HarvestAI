"""Week-3 smoke test: verifies the plumbing works end-to-end without
actually hitting Sentinel Hub.

What it checks:
1. Backend imports cleanly
2. Redis is reachable
3. RQ dispatcher enqueues a fake job
4. PU tracker is callable
5. Quota and observation endpoints exist and require auth

Run with: uv run python scripts/smoke_test_week3.py
"""
from __future__ import annotations

import asyncio
import secrets

from httpx import ASGITransport, AsyncClient


async def main() -> None:
    print("[1/6] Importing app...")
    from app.main import app
    from app.redis_clients import get_async_redis, get_sync_redis

    print("[2/6] Pinging Redis (sync + async)...")
    sync = get_sync_redis()
    sync.ping()
    print("       sync OK")
    a = get_async_redis()
    await a.ping()
    print("       async OK")

    print("[3/6] Testing PU tracker increment...")
    from app.integrations.sentinel_hub.pu_tracker import PuTracker

    tracker = PuTracker(a, monthly_limit=10000)
    before = float(await tracker.current_usage())
    await tracker.record(0.001)  # tiny canary increment
    after = float(await tracker.current_usage())
    assert after > before
    print(f"       PU counter incremented: {before} -> {after}")

    print("[4/6] Testing RQ dispatcher (enqueue + cancel)...")
    from app.workers.dispatcher import enqueue_fetch_observations
    from rq.job import Job

    # Enqueue against a fake field id; never let it run.
    handle = enqueue_fetch_observations(field_id=99_999_999, years_back=1)
    print(f"       Enqueued job_id={handle.job_id}")
    job = Job.fetch(handle.job_id, connection=sync)
    job.cancel()
    print(f"       Job state after cancel: {job.get_status()}")

    print("[5/6] Testing API endpoints (require auth -> 401)...")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        for path in (
            "/api/quota",
            "/api/fields/1/observations",
            "/api/fields/1/observations/refresh",
            "/api/jobs/abc",
        ):
            r = await c.get(path) if path == "/api/quota" or path.endswith("observations") or path.startswith("/api/jobs") else await c.post(path)
            assert r.status_code == 401, f"Expected 401 for {path}, got {r.status_code}"
        print("       All endpoints correctly require auth")

    print("[6/6] Register + login + check quota...")
    email = f"smoke3+{secrets.token_hex(4)}@example.com"
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post(
            "/api/auth/register",
            json={"email": email, "password": "very-strong-pass"},
        )
        assert r.status_code == 201
        r = await c.post(
            "/api/auth/login",
            json={"email": email, "password": "very-strong-pass"},
        )
        token = r.json()["access_token"]
        r = await c.get("/api/quota", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200, r.text
        q = r.json()
        print(
            f"       Quota: {q['units_used']:.3f} / {q['units_limit']:.0f} PU "
            f"({q['percent_used']:.2f}%)"
        )

    print("\n[ALL OK] Week 3 backend infra is working")


if __name__ == "__main__":
    asyncio.run(main())
