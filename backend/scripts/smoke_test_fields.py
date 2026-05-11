"""End-to-end smoke test against a live PostgreSQL+PostGIS instance.

Not a pytest — run with `uv run python scripts/smoke_test_fields.py`.
Exercises the full FastAPI app via httpx ASGI transport, hitting the real
database via the configured DATABASE_URL. Verifies CRUD + generated columns.
"""
from __future__ import annotations

import asyncio
import secrets

from httpx import ASGITransport, AsyncClient

from app.main import app


async def main() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        email = f"smoke+{secrets.token_hex(4)}@example.com"
        pwd = "smoke-pass-12345"

        # 1. Register + login
        r = await c.post("/api/auth/register", json={"email": email, "password": pwd})
        assert r.status_code == 201, r.text
        r = await c.post("/api/auth/login", json={"email": email, "password": pwd})
        assert r.status_code == 200, r.text
        token = r.json()["access_token"]
        h = {"Authorization": f"Bearer {token}"}

        # 2. Empty list
        r = await c.get("/api/fields", headers=h)
        assert r.status_code == 200
        assert r.json() == []
        print("[OK] Empty fields list")

        # 3. Create a ~2 ha field near Lviv (wheat)
        payload = {
            "name": "Південне поле №1",
            "crop_type": "wheat",
            "season_year": 2026,
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [24.030, 49.840],
                        [24.0320, 49.840],
                        [24.0320, 49.8410],
                        [24.030, 49.8410],
                        [24.030, 49.840],
                    ]
                ],
            },
        }
        r = await c.post("/api/fields", json=payload, headers=h)
        assert r.status_code == 201, r.text
        created = r.json()
        print(
            f"[OK] Created field id={created['id']} "
            f"area={created['area_ha']} ha "
            f"centroid={created['centroid']['coordinates']} "
            f"color={created['color']}"
        )
        assert 0.5 < created["area_ha"] < 10.0
        assert created["color"] == "#f0c419"  # default wheat color
        fid = created["id"]

        # 4. List
        r = await c.get("/api/fields", headers=h)
        assert r.status_code == 200
        assert len(r.json()) == 1
        print("[OK] List has 1 field")

        # 5. Get single
        r = await c.get(f"/api/fields/{fid}", headers=h)
        assert r.status_code == 200
        print("[OK] Single field fetched")

        # 6. Update — change name and crop_type
        r = await c.patch(
            f"/api/fields/{fid}",
            json={"name": "Перейменоване поле", "crop_type": "corn"},
            headers=h,
        )
        assert r.status_code == 200, r.text
        updated = r.json()
        assert updated["name"] == "Перейменоване поле"
        assert updated["crop_type"] == "corn"
        print(f"[OK] Updated field, color now {updated['color']}")

        # 7. Update geometry (smaller polygon)
        smaller = {
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [24.030, 49.840],
                        [24.0310, 49.840],
                        [24.0310, 49.8410],
                        [24.030, 49.8410],
                        [24.030, 49.840],
                    ]
                ],
            }
        }
        r = await c.patch(f"/api/fields/{fid}", json=smaller, headers=h)
        assert r.status_code == 200, r.text
        new_area = r.json()["area_ha"]
        print(f"[OK] Geometry updated, new area={new_area} ha (smaller than before)")
        assert new_area < float(created["area_ha"])

        # 8. Cross-user 404
        other_email = f"other+{secrets.token_hex(4)}@example.com"
        await c.post("/api/auth/register", json={"email": other_email, "password": pwd})
        r = await c.post(
            "/api/auth/login", json={"email": other_email, "password": pwd}
        )
        other_token = r.json()["access_token"]
        r = await c.get(
            f"/api/fields/{fid}", headers={"Authorization": f"Bearer {other_token}"}
        )
        assert r.status_code == 404, r.text
        print("[OK] Cross-user access returns 404 (no leak)")

        # 9. Delete
        r = await c.delete(f"/api/fields/{fid}", headers=h)
        assert r.status_code == 204
        r = await c.get("/api/fields", headers=h)
        assert r.json() == []
        print("[OK] Delete works")

        # 10. Validation: too small polygon
        bad = {
            **payload,
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [24.030, 49.840],
                        [24.0301, 49.840],
                        [24.0301, 49.8401],
                        [24.030, 49.8401],
                        [24.030, 49.840],
                    ]
                ],
            },
        }
        r = await c.post("/api/fields", json=bad, headers=h)
        assert r.status_code == 400, r.text
        print(f"[OK] Validation rejects too-small: {r.json()['detail']}")

        print("\n[ALL OK] Smoke checks passed")


if __name__ == "__main__":
    asyncio.run(main())
