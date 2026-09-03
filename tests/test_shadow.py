from modelgate.serving import metrics
from modelgate.serving.shadow import ShadowRecord, ShadowTracker
from tests.conftest import ADMIN, GOOD_INPUT


def _rec(primary: float, shadow: float) -> ShadowRecord:
    return ShadowRecord("r", "v1", "v2", primary, shadow, at=0.0)


def test_tracker_report_math():
    t = ShadowTracker(threshold_minutes=2.0, max_records=10)
    for p, s in [(10, 10.5), (20, 23), (30, 29), (40, 46)]:
        t.record(_rec(p, s))
    rep = t.report()
    assert rep["count"] == 4 and rep["window"] == 4 and rep["errors"] == 0
    assert rep["abs_delta_minutes"]["mean"] == 2.625
    assert rep["abs_delta_minutes"]["max"] == 6.0
    assert rep["abs_delta_minutes"]["p50"] == 1.0
    assert rep["abs_delta_minutes"]["p95"] == 6.0
    assert rep["share_beyond_threshold"] == 0.5
    assert rep["shadow_bias_minutes"] == 2.125
    assert rep["last"]["shadow"] == 46


def test_tracker_window_is_bounded_but_count_is_not():
    t = ShadowTracker(threshold_minutes=1.0, max_records=3)
    for i in range(10):
        t.record(_rec(10, 10 + i))
    rep = t.report()
    assert rep["count"] == 10 and rep["window"] == 3
    assert rep["abs_delta_minutes"]["max"] == 9.0
    t.record_error()
    assert t.report()["errors"] == 1
    t.reset()
    rep = t.report()
    assert (rep["count"], rep["window"], rep["errors"]) == (0, 0, 0)


def test_empty_report_is_well_formed():
    rep = ShadowTracker(2.0, 5).report()
    assert rep["count"] == 0 and rep["last"] is None
    assert rep["abs_delta_minutes"]["p95"] == 0.0


def test_shadow_runs_record_divergence_and_client_gets_primary(client):
    r = client.post("/admin/shadow", json={"version": "v2"}, headers=ADMIN)
    assert r.status_code == 200 and r.json() == {"shadow": "v2"}

    before = metrics.histogram_count(metrics.SHADOW_DIVERGENCE, primary="v1", shadow="v2")
    shadow_ok = metrics.counter_value(metrics.SHADOW_REQUESTS, shadow="v2", outcome="ok")
    n = 25
    versions = {client.post("/predict", json=GOOD_INPUT).json()["model_version"] for _ in range(n)}
    assert versions == {"v1"}

    assert metrics.histogram_count(metrics.SHADOW_DIVERGENCE, primary="v1", shadow="v2") == (
        before + n
    )
    assert metrics.counter_value(metrics.SHADOW_REQUESTS, shadow="v2", outcome="ok") == (
        shadow_ok + n
    )

    rep = client.get("/admin/shadow/report", headers=ADMIN).json()
    assert rep["primary"] == "v1" and rep["shadow"] == "v2"
    assert rep["count"] == n and rep["errors"] == 0
    # v1 and v2 disagree on this trip; the report should show a real, finite delta.
    assert 0 < rep["abs_delta_minutes"]["mean"] < 30
    assert rep["abs_delta_minutes"]["p95"] >= rep["abs_delta_minutes"]["p50"]
    assert 0 <= rep["share_beyond_threshold"] <= 1
    assert rep["last"]["request_id"]


def test_shadow_can_be_cleared_and_stops_recording(client):
    client.post("/admin/shadow", json={"version": "v2"}, headers=ADMIN)
    client.post("/predict", json=GOOD_INPUT)
    r = client.post("/admin/shadow", json={"version": None}, headers=ADMIN)
    assert r.json() == {"shadow": None}
    before = metrics.counter_value(metrics.SHADOW_REQUESTS, shadow="v2", outcome="ok")
    client.post("/predict", json=GOOD_INPUT)
    assert metrics.counter_value(metrics.SHADOW_REQUESTS, shadow="v2", outcome="ok") == before
    assert client.get("/admin/shadow/report", headers=ADMIN).json()["count"] == 0


def test_shadow_cannot_equal_primary(client):
    assert client.post("/admin/shadow", json={"version": "v1"}, headers=ADMIN).status_code == 409


def test_shadow_failure_never_reaches_client(app, client):
    class Broken:
        version = "v2"

        def predict_batch(self, _):
            raise RuntimeError("shadow exploded")

    app.state.registry._shadow = Broken()
    errors = metrics.counter_value(metrics.SHADOW_REQUESTS, shadow="v2", outcome="error")
    dropped = metrics.counter_value(metrics.DROPPED_REQUESTS)
    r = client.post("/predict", json=GOOD_INPUT)
    assert r.status_code == 200 and r.json()["model_version"] == "v1"
    assert metrics.counter_value(metrics.SHADOW_REQUESTS, shadow="v2", outcome="error") == (
        errors + 1
    )
    assert metrics.counter_value(metrics.DROPPED_REQUESTS) == dropped
    assert client.get("/admin/shadow/report", headers=ADMIN).json()["errors"] == 1
