import json

import pytest

from modelgate.serving import metrics
from tests.conftest import GOOD_INPUT

JSON = {"content-type": "application/json"}


def _rejections(reason: str) -> float:
    return metrics.counter_value(metrics.INPUT_REJECTIONS, reason=reason)


def test_valid_input_returns_plausible_eta(client):
    r = client.post("/predict", json=GOOD_INPUT)
    assert r.status_code == 200
    body = r.json()
    assert body["model_version"] == "v1"
    assert len(body["request_id"]) == 16
    # 8.2 km in heavy traffic and rain: somewhere between 10 and 60 minutes.
    assert 10 < body["eta_minutes"] < 60


def test_request_id_header_is_echoed(client):
    r = client.post("/predict", json=GOOD_INPUT, headers={"X-Request-ID": "abc-123"})
    assert r.json()["request_id"] == "abc-123"


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("distance_km", -1, "out_of_range"),
        ("distance_km", 500.5, "out_of_range"),
        ("hour_of_day", 24, "out_of_range"),
        ("hour_of_day", -1, "out_of_range"),
        ("day_of_week", 7, "out_of_range"),
        ("traffic_index", 1.01, "out_of_range"),
        ("traffic_index", -0.1, "out_of_range"),
        ("pickup_zone_id", 99, "unknown_zone"),
        ("pickup_zone_id", 0, "unknown_zone"),
        ("hour_of_day", "5", "wrong_type"),
        ("hour_of_day", 5.5, "wrong_type"),
        ("distance_km", "8.2", "wrong_type"),
        ("is_raining", 1, "wrong_type"),
        ("is_raining", "yes", "wrong_type"),
        ("pickup_zone_id", True, "wrong_type"),
    ],
)
def test_each_rule_rejects_with_422_and_counts(client, field, value, reason):
    before = _rejections(reason)
    r = client.post("/predict", json={**GOOD_INPUT, field: value})
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "invalid input"
    assert body["rejections"][0]["field"] == field
    assert body["rejections"][0]["reason"] == reason
    assert _rejections(reason) == before + 1


@pytest.mark.parametrize("bad", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_floats_are_rejected(client, bad):
    before = _rejections("not_finite")
    raw = json.dumps({**GOOD_INPUT, "distance_km": 1.0}).replace("1.0", bad)
    r = client.post("/predict", content=raw, headers=JSON)
    assert r.status_code == 422
    assert r.json()["rejections"][0]["reason"] == "not_finite"
    assert _rejections("not_finite") == before + 1


def test_unknown_field_is_rejected(client):
    before = _rejections("unknown_field")
    r = client.post("/predict", json={**GOOD_INPUT, "rider_id": 42})
    assert r.status_code == 422
    assert r.json()["rejections"][0]["field"] == "rider_id"
    assert _rejections("unknown_field") == before + 1


def test_missing_field_is_rejected(client):
    before = _rejections("missing_field")
    payload = dict(GOOD_INPUT)
    del payload["traffic_index"]
    r = client.post("/predict", json=payload)
    assert r.status_code == 422
    assert r.json()["rejections"][0]["field"] == "traffic_index"
    assert _rejections("missing_field") == before + 1


def test_malformed_json_is_rejected(client):
    before = _rejections("malformed_body")
    r = client.post("/predict", content=b"{not json", headers=JSON)
    assert r.status_code == 422
    assert r.json()["rejections"][0]["reason"] == "malformed_body"
    assert _rejections("malformed_body") == before + 1


def test_rejections_count_as_rejected_requests_not_drops(client):
    dropped = metrics.counter_value(metrics.DROPPED_REQUESTS)
    rejected = metrics.counter_value(metrics.REQUESTS, version="v1", outcome="rejected")
    client.post("/predict", json={**GOOD_INPUT, "hour_of_day": 99})
    assert metrics.counter_value(metrics.DROPPED_REQUESTS) == dropped
    assert metrics.counter_value(metrics.REQUESTS, version="v1", outcome="rejected") == rejected + 1


def test_boundary_values_are_accepted(client):
    edge = {
        "distance_km": 500.0,
        "hour_of_day": 23,
        "day_of_week": 6,
        "pickup_zone_id": 12,
        "traffic_index": 1.0,
        "is_raining": False,
    }
    assert client.post("/predict", json=edge).status_code == 200
    zero = {**edge, "distance_km": 0, "hour_of_day": 0, "day_of_week": 0, "traffic_index": 0}
    assert client.post("/predict", json=zero).status_code == 200
