"""Canary routing: the split holds, rollback fires, and nothing is dropped."""

from __future__ import annotations

import time
from collections import Counter

from modelgate.serving import metrics
from modelgate.serving.canary import CanaryRouter, CanaryThresholds
from tests.conftest import ADMIN, GOOD_INPUT


class _Fake:
    def __init__(self, version: str) -> None:
        self.version = version


def _thresholds(**overrides) -> CanaryThresholds:
    base = dict(window_seconds=60.0, min_samples=20, error_rate_delta=0.05, latency_ratio=2.0)
    base.update(overrides)
    return CanaryThresholds(**base)


def test_router_split_is_exact_for_weight():
    router = CanaryRouter(_thresholds())
    primary, candidate = _Fake("v1"), _Fake("v2")
    router.start(candidate, 0.05)
    picks = Counter("v2" if router.pick(primary) is candidate else "v1" for _ in range(2000))
    assert picks == {"v1": 1900, "v2": 100}
    router.start(candidate, 0.25)
    picks = Counter("v2" if router.pick(primary) is candidate else "v1" for _ in range(400))
    assert picks == {"v1": 300, "v2": 100}


def test_router_never_routes_to_candidate_equal_to_primary_or_when_cleared():
    router = CanaryRouter(_thresholds())
    v1 = _Fake("v1")
    router.start(v1, 1.0)
    assert router.pick(v1) is None
    router.start(_Fake("v2"), 1.0)
    router.clear()
    assert router.pick(v1) is None and router.weight == 0.0 and router.status == "none"


def test_router_rolls_back_on_error_rate():
    router = CanaryRouter(_thresholds(min_samples=10))
    router.start(_Fake("v2"), 0.5)
    for _ in range(10):
        router.record("v1", 0.001, True, "v1")
    for i in range(10):
        router.record("v2", 0.001, i % 2 == 0, "v1")
    assert router.status == "rolled_back" and router.candidate is None
    rb = router.report("v1")["last_rollback"]
    assert rb["reason"] == "error_rate" and rb["version"] == "v2" and rb["candidate"] == 0.5


def test_router_rolls_back_on_p95_latency():
    router = CanaryRouter(_thresholds(min_samples=10, latency_floor_ms=1.0))
    router.start(_Fake("v2"), 0.5)
    for _ in range(10):
        router.record("v1", 0.001, True, "v1")
    for _ in range(10):
        router.record("v2", 0.010, True, "v1")
    rb = router.report("v1")["last_rollback"]
    assert rb is not None and rb["reason"] == "latency"
    assert rb["candidate"] == 10.0 and rb["primary"] == 1.0


def test_router_latency_floor_ignores_sub_millisecond_noise():
    router = CanaryRouter(_thresholds(min_samples=10, latency_floor_ms=1.0))
    router.start(_Fake("v2"), 0.5)
    for _ in range(10):
        router.record("v1", 0.0002, True, "v1")
    for _ in range(10):
        router.record("v2", 0.0008, True, "v1")  # 4x slower but 0.6 ms apart
    assert router.status == "active"


def test_router_window_expires_old_samples():
    now = [1000.0]
    router = CanaryRouter(_thresholds(window_seconds=10.0, min_samples=5), clock=lambda: now[0])
    router.start(_Fake("v2"), 0.5)
    for _ in range(5):
        router.record("v1", 0.001, True, "v1")
        router.record("v2", 0.001, False, "v1")
    # Not enough samples yet to judge: 5 candidate samples arrive interleaved; the fifth
    # candidate record sees 5 and 5 and fires.
    assert router.status == "rolled_back"
    router.start(_Fake("v2"), 0.5)
    for _ in range(4):
        router.record("v2", 0.001, False, "v1")
    now[0] += 30.0
    for _ in range(5):
        router.record("v1", 0.001, True, "v1")
    router.record("v2", 0.001, False, "v1")
    # The four old failures left the window, so only one recent sample exists: no verdict.
    assert router.status == "active"
    assert router.report("v1")["versions"]["v2"]["samples"] == 1


def test_canary_split_holds_over_api_and_nothing_drops(client):
    dropped = metrics.counter_value(metrics.DROPPED_REQUESTS)
    r = client.post("/admin/canary", json={"version": "v2", "weight": 0.2}, headers=ADMIN)
    assert r.status_code == 200 and r.json() == {
        "candidate": "v2",
        "weight": 0.2,
        "status": "active",
    }
    assert metrics.gauge_value(metrics.CANARY_WEIGHT) == 0.2

    n = 500
    statuses, versions = Counter(), Counter()
    for i in range(n):
        resp = client.post("/predict", json={**GOOD_INPUT, "distance_km": 1 + (i % 25)})
        statuses[resp.status_code] += 1
        if resp.status_code == 200:
            versions[resp.json()["model_version"]] += 1
    assert statuses == {200: n}
    assert versions == {"v1": 400, "v2": 100}
    assert metrics.counter_value(metrics.DROPPED_REQUESTS) == dropped

    rep = client.get("/admin/canary/report", headers=ADMIN).json()
    assert rep["status"] == "active" and rep["candidate"] == "v2" and rep["primary"] == "v1"
    assert rep["versions"]["v2"]["samples"] == 100 and rep["versions"]["v1"]["samples"] == 400
    assert rep["versions"]["v2"]["error_rate"] == 0.0 and rep["rollbacks"] == 0

    text = client.get("/metrics").text
    assert 'modelgate_canary_info{version="v2"} 1.0' in text
    assert "modelgate_canary_weight 0.2" in text


def test_broken_canary_rolls_back_without_dropping_a_request(app, client):
    class Broken:
        version = "v2"

        def predict_batch(self, _):
            raise RuntimeError("candidate exploded")

    dropped = metrics.counter_value(metrics.DROPPED_REQUESTS)
    rollbacks = metrics.counter_value(metrics.CANARY_ROLLBACKS, reason="error_rate")
    app.state.canary.thresholds = CanaryThresholds(min_samples=10, error_rate_delta=0.05)
    app.state.canary.start(Broken(), 0.5)

    versions = []
    for _ in range(60):
        r = client.post("/predict", json=GOOD_INPUT)
        assert r.status_code == 200
        versions.append(r.json()["model_version"])
    # Every answer came from v1: canary failures fall back, and after rollback nothing
    # routes to v2 at all.
    assert set(versions) == {"v1"}
    assert metrics.counter_value(metrics.DROPPED_REQUESTS) == dropped
    assert metrics.counter_value(metrics.CANARY_ROLLBACKS, reason="error_rate") == rollbacks + 1

    rep = client.get("/admin/canary/report", headers=ADMIN).json()
    assert rep["status"] == "rolled_back" and rep["candidate"] is None and rep["weight"] == 0.0
    assert rep["last_rollback"]["reason"] == "error_rate"
    assert rep["last_rollback"]["candidate"] == 1.0 and rep["last_rollback"]["primary"] == 0.0
    # The verdict fired at the sample floor: exactly 10 candidate requests were tried.
    assert rep["versions"]["v2"]["samples"] == 10 and rep["versions"]["v2"]["errors"] == 10
    assert metrics.gauge_value(metrics.CANARY_INFO, version="v2") == 0.0


def test_slow_canary_rolls_back_on_latency(app, client):
    real = app.state.registry.load("v2")

    class Slow:
        version = "v2"

        def predict_batch(self, features):
            time.sleep(0.004)
            return real.predict_batch(features)

    rollbacks = metrics.counter_value(metrics.CANARY_ROLLBACKS, reason="latency")
    app.state.canary.thresholds = CanaryThresholds(min_samples=10, latency_ratio=2.0)
    app.state.canary.start(Slow(), 0.5)
    for _ in range(40):
        assert client.post("/predict", json=GOOD_INPUT).status_code == 200
    rep = client.get("/admin/canary/report", headers=ADMIN).json()
    assert rep["status"] == "rolled_back" and rep["last_rollback"]["reason"] == "latency"
    assert rep["last_rollback"]["candidate"] > rep["last_rollback"]["primary"] * 2
    assert metrics.counter_value(metrics.CANARY_ROLLBACKS, reason="latency") == rollbacks + 1


def test_healthy_canary_stays_active(app, client):
    # v2 is a wider net than v1, so a few milliseconds of inference make its p95 exceed twice the
    # primary's on a loaded runner even when nothing is wrong: that is what latency_floor_ms is for,
    # and it is lifted here above anything a shared runner produces. The guard itself is asserted in
    # test_slow_canary_rolls_back_on_latency, which injects a delay rather than trusting the host.
    app.state.canary.thresholds = CanaryThresholds(min_samples=10, latency_floor_ms=250.0)
    client.post("/admin/canary", json={"version": "v2", "weight": 0.5}, headers=ADMIN)
    for i in range(200):
        assert (
            client.post("/predict", json={**GOOD_INPUT, "distance_km": 1 + i % 30}).status_code
            == 200
        )
    rep = client.get("/admin/canary/report", headers=ADMIN).json()
    assert rep["status"] == "active" and rep["rollbacks"] == 0
    assert rep["versions"]["v2"]["samples"] == 100


def test_canary_admin_validation(client):
    assert client.post("/admin/canary", json={"version": "v2"}).status_code == 401
    r = client.post("/admin/canary", json={"version": "v9", "weight": 0.1}, headers=ADMIN)
    assert r.status_code == 404
    r = client.post("/admin/canary", json={"version": "v1", "weight": 0.1}, headers=ADMIN)
    assert r.status_code == 409
    for weight in (0.0, 1.5, -0.1, "0.1"):
        r = client.post("/admin/canary", json={"version": "v2", "weight": weight}, headers=ADMIN)
        assert r.status_code == 422, weight
    r = client.post("/admin/canary", json={"version": "v2", "weight": 1.0}, headers=ADMIN)
    assert r.status_code == 200
    assert client.post("/predict", json=GOOD_INPUT).json()["model_version"] == "v2"
    r = client.post("/admin/canary", json={"version": None}, headers=ADMIN)
    assert r.json() == {"candidate": None, "weight": 0.0, "status": "none"}
    assert client.post("/predict", json=GOOD_INPUT).json()["model_version"] == "v1"


def test_promoting_the_canary_clears_it(app, client):
    client.post("/admin/canary", json={"version": "v2", "weight": 0.5}, headers=ADMIN)
    r = client.post("/admin/promote", json={"version": "v2"}, headers=ADMIN)
    assert r.status_code == 200
    rep = client.get("/admin/canary/report", headers=ADMIN).json()
    assert rep["candidate"] is None and rep["primary"] == "v2" and rep["weight"] == 0.0
    assert client.post("/predict", json=GOOD_INPUT).json()["model_version"] == "v2"
