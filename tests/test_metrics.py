from tests.conftest import ADMIN, GOOD_INPUT

EXPECTED_SERIES = [
    "modelgate_requests_total",
    "modelgate_request_latency_seconds_bucket",
    "modelgate_predictions_eta_minutes_bucket",
    "modelgate_input_rejections_total",
    "modelgate_shadow_divergence_minutes_bucket",
    "modelgate_shadow_requests_total",
    "modelgate_model_version_info",
    "modelgate_version_swaps_total",
    "modelgate_dropped_requests_total",
    "modelgate_models_loaded",
]


def test_metrics_endpoint_exposes_expected_series(client):
    client.post("/admin/shadow", json={"version": "v2"}, headers=ADMIN)
    client.post("/predict", json=GOOD_INPUT)
    client.post("/predict", json={**GOOD_INPUT, "hour_of_day": 30})
    client.post("/admin/promote", json={"version": "v2"}, headers=ADMIN)

    r = client.get("/metrics")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    text = r.text
    for name in EXPECTED_SERIES:
        assert f"\n{name}" in text or text.startswith(name), name

    lines = text.splitlines()
    assert any(l.startswith('modelgate_requests_total{outcome="ok",version="v1"}') for l in lines)
    assert any(
        l.startswith('modelgate_input_rejections_total{reason="out_of_range"}') for l in lines
    )
    assert 'modelgate_model_version_info{role="primary",version="v2"} 1.0' in lines
    assert 'modelgate_model_version_info{role="primary",version="v1"} 0.0' in lines
    assert "modelgate_dropped_requests_total 0.0" in lines
    assert not any(l.startswith("modelgate_requests_created") for l in lines)
