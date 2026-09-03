"""Warm pool: pre-warmed promotes swap without loading, eviction spares pinned roles."""

from __future__ import annotations

import asyncio
import json
import shutil
from collections import Counter
from pathlib import Path

import httpx
import torch
from fastapi.testclient import TestClient

from modelgate.model.features import FEATURE_DIM
from modelgate.serving import metrics
from modelgate.serving.app import create_app
from modelgate.serving.registry import ModelRegistry
from tests.conftest import ADMIN, ARTIFACTS, GOOD_INPUT, make_settings


def _four_version_artifacts(tmp_path: Path) -> Path:
    """A manifest with v1..v4 where v3 and v4 reuse the v1 and v2 weights."""
    manifest = json.loads((ARTIFACTS / "manifest.json").read_text())
    for src in ("eta_v1.pt", "eta_v2.pt"):
        shutil.copy(ARTIFACTS / src, tmp_path / src)
    manifest["versions"]["v3"] = {**manifest["versions"]["v1"]}
    manifest["versions"]["v4"] = {**manifest["versions"]["v2"]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


def test_warm_loads_once_and_second_call_is_a_hit():
    reg = ModelRegistry(ARTIFACTS, pool_size=3)
    misses = metrics.counter_value(metrics.WARM_LOADS, hit="false")
    hits = metrics.counter_value(metrics.WARM_LOADS, hit="true")
    first = reg.warm("v2")
    assert first["loaded"] is True and first["load_seconds"] > 0
    assert first["resident"] == ["v2"] and first["evicted"] == []
    second = reg.warm("v2")
    assert second["loaded"] is False and second["load_seconds"] == 0.0
    assert metrics.counter_value(metrics.WARM_LOADS, hit="false") == misses + 1
    assert metrics.counter_value(metrics.WARM_LOADS, hit="true") == hits + 1


def test_promote_after_warm_is_prewarmed_and_does_not_load():
    reg = ModelRegistry(ARTIFACTS, pool_size=3)
    cold = reg.promote("v1")
    assert cold["prewarmed"] is False and cold["load_seconds"] > 0
    reg.warm("v2")
    warm_obj = reg.load("v2")
    hot = reg.promote("v2")
    assert hot["prewarmed"] is True and hot["load_seconds"] == 0.0
    assert reg.primary is warm_obj
    assert reg.describe()["pool"] == {"size": 3, "resident": ["v1", "v2"], "pinned": ["v1", "v2"]}


def test_eviction_drops_least_recently_touched_unpinned(tmp_path):
    reg = ModelRegistry(_four_version_artifacts(tmp_path), pool_size=2)
    evictions = metrics.counter_value(metrics.MODEL_EVICTIONS)
    reg.promote("v1")
    reg.warm("v2")
    assert reg.resident_versions() == ["v1", "v2"]
    out = reg.warm("v3")
    assert out["evicted"] == ["v2"] and reg.resident_versions() == ["v1", "v3"]
    reg.promote("v3")  # v1 becomes previous: pinned as the rollback target
    out = reg.warm("v4")
    assert out["evicted"] == [] and reg.resident_versions() == ["v1", "v3", "v4"]
    assert metrics.counter_value(metrics.MODEL_EVICTIONS) == evictions + 1
    assert metrics.gauge_value(metrics.MODEL_LOADED) == 3


def test_extra_pins_protect_a_version(tmp_path):
    pins: set[str] = set()
    reg = ModelRegistry(_four_version_artifacts(tmp_path), pool_size=1, extra_pins=lambda: pins)
    reg.promote("v1")
    pins.add("v2")
    assert reg.warm("v2")["evicted"] == []
    # v3 is protected while it is being warmed; v1 (primary) and v2 (pinned) cannot go,
    # so the pool stays over capacity rather than dropping a role holder.
    assert reg.warm("v3")["evicted"] == [] and reg.resident_versions() == ["v1", "v2", "v3"]
    pins.clear()
    out = reg.warm("v4")
    assert out["evicted"] == ["v2", "v3"] and reg.resident_versions() == ["v1", "v4"]


def test_evicted_model_still_answers_in_flight_requests(tmp_path):
    reg = ModelRegistry(_four_version_artifacts(tmp_path), pool_size=1)
    reg.promote("v1")
    held = reg.load("v2")
    reg.warm("v3")
    assert not reg.is_resident("v2")
    assert held.predict(torch.zeros(1, FEATURE_DIM)) > 0


def test_warm_endpoint_and_versions_report_pool(client):
    assert client.post("/admin/warm", json={"version": "v2"}).status_code == 401
    assert client.post("/admin/warm", json={"version": "v9"}, headers=ADMIN).status_code == 404
    r = client.post("/admin/warm", json={"version": "v2"}, headers=ADMIN)
    assert r.status_code == 200 and r.json()["loaded"] is True
    assert r.json()["resident"] == ["v1", "v2"]
    info = client.get("/admin/versions", headers=ADMIN).json()
    assert info["pool"]["resident"] == ["v1", "v2"] and info["pool"]["size"] == 3
    assert {v["version"]: v["loaded"] for v in info["versions"]} == {"v1": True, "v2": True}
    r = client.post("/admin/promote", json={"version": "v2"}, headers=ADMIN)
    assert r.json()["prewarmed"] is True and r.json()["load_seconds"] == 0.0
    assert client.post("/predict", json=GOOD_INPUT).json()["model_version"] == "v2"
    swaps = client.get("/admin/versions", headers=ADMIN).json()["swaps"]
    assert swaps[-1]["prewarmed"] is True
    text = client.get("/metrics").text
    assert 'modelgate_swap_load_seconds_count{prewarmed="true"}' in text
    assert "modelgate_model_pool_slots 3.0" in text


def test_warm_versions_setting_preloads_at_startup():
    app = create_app(make_settings(warm_versions=("v2", "v9")))
    with TestClient(app) as c:
        info = c.get("/admin/versions", headers=ADMIN).json()
        assert info["pool"]["resident"] == ["v1", "v2"]
        assert c.post("/admin/promote", json={"version": "v2"}, headers=ADMIN).json()["prewarmed"]


async def test_prewarmed_swap_under_load_drops_nothing():
    app = create_app(make_settings())
    total, workers = 1500, 50
    dropped = metrics.counter_value(metrics.DROPPED_REQUESTS)
    outcomes: list[tuple[int, str | None]] = []
    done = 0
    swap_record: dict | None = None

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            sem = asyncio.Semaphore(workers)

            async def one(i: int) -> None:
                nonlocal done
                async with sem:
                    await asyncio.sleep(0)
                    r = await client.post(
                        "/predict", json={**GOOD_INPUT, "distance_km": 1 + (i % 40) * 0.5}
                    )
                outcomes.append((r.status_code, r.json().get("model_version")))
                done += 1

            async def controller() -> None:
                nonlocal swap_record
                while done < total // 4:
                    await asyncio.sleep(0.001)
                r = await client.post("/admin/warm", json={"version": "v2"}, headers=ADMIN)
                assert r.status_code == 200 and r.json()["loaded"] is True
                while done < total // 2:
                    await asyncio.sleep(0.001)
                r = await client.post("/admin/promote", json={"version": "v2"}, headers=ADMIN)
                assert r.status_code == 200
                swap_record = r.json()

            await asyncio.gather(controller(), *(one(i) for i in range(total)))

    assert Counter(s for s, _ in outcomes) == {200: total}
    assert metrics.counter_value(metrics.DROPPED_REQUESTS) == dropped
    assert swap_record is not None and swap_record["prewarmed"] is True
    assert swap_record["load_seconds"] == 0.0
    versions = Counter(v for _, v in outcomes)
    assert versions["v1"] > 0 and versions["v2"] > 0
