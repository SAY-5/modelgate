"""Version swaps under concurrent load must not drop a single request."""

from __future__ import annotations

import asyncio
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import httpx
from fastapi.testclient import TestClient

from modelgate.serving import metrics
from modelgate.serving.app import create_app
from tests.conftest import ADMIN, GOOD_INPUT, make_settings

TOTAL = 2000
WORKERS = 50


async def test_async_load_with_mid_run_promote():
    app = create_app(make_settings())
    dropped_before = metrics.counter_value(metrics.DROPPED_REQUESTS)
    outcomes: list[tuple[int, str | None]] = []
    done = 0
    swap_seen_at: int | None = None

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            sem = asyncio.Semaphore(WORKERS)

            async def one(i: int) -> None:
                nonlocal done
                payload = {**GOOD_INPUT, "distance_km": 1 + (i % 40) * 0.5}
                async with sem:
                    r = await client.post("/predict", json=payload)
                version = r.json().get("model_version") if r.status_code == 200 else None
                outcomes.append((r.status_code, version))
                done += 1

            async def promoter() -> None:
                nonlocal swap_seen_at
                while done < TOTAL // 2:
                    await asyncio.sleep(0.001)
                r = await client.post("/admin/promote", json={"version": "v2"}, headers=ADMIN)
                assert r.status_code == 200 and r.json()["to"] == "v2"
                swap_seen_at = done

            await asyncio.gather(promoter(), *(one(i) for i in range(TOTAL)))

    statuses = Counter(s for s, _ in outcomes)
    assert statuses == {200: TOTAL}, statuses
    assert metrics.counter_value(metrics.DROPPED_REQUESTS) == dropped_before
    versions = Counter(v for _, v in outcomes)
    assert set(versions) == {"v1", "v2"}, versions
    assert versions["v1"] > 0 and versions["v2"] > 0
    assert swap_seen_at is not None and 0 < swap_seen_at < TOTAL
    # Once the swap has happened, nothing goes back to v1 (no torn or stale reads).
    tail = [v for _, v in outcomes[-100:]]
    assert set(tail) == {"v2"}


def test_threaded_load_with_mid_run_promote():
    app = create_app(make_settings())
    dropped_before = metrics.counter_value(metrics.DROPPED_REQUESTS)
    outcomes: list[tuple[int, str | None]] = []
    lock = threading.Lock()
    swap_trigger = threading.Event()

    with TestClient(app) as client:

        def one(i: int) -> None:
            r = client.post("/predict", json={**GOOD_INPUT, "distance_km": 2 + (i % 30)})
            version = r.json().get("model_version") if r.status_code == 200 else None
            with lock:
                outcomes.append((r.status_code, version))
                if len(outcomes) >= TOTAL // 2:
                    swap_trigger.set()

        def promoter() -> None:
            swap_trigger.wait(timeout=60)
            r = client.post("/admin/promote", json={"version": "v2"}, headers=ADMIN)
            assert r.status_code == 200

        with ThreadPoolExecutor(max_workers=WORKERS + 1) as pool:
            swap = pool.submit(promoter)
            futures = [pool.submit(one, i) for i in range(TOTAL)]
            for f in futures:
                f.result()
            swap.result()

    statuses = Counter(s for s, _ in outcomes)
    assert statuses == {200: TOTAL}, statuses
    assert metrics.counter_value(metrics.DROPPED_REQUESTS) == dropped_before
    versions = Counter(v for _, v in outcomes)
    assert versions["v1"] > 0 and versions["v2"] > 0, versions
