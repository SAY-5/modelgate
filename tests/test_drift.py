"""Feature drift: shifted inputs score high, training-like traffic does not."""

from __future__ import annotations

import json
from decimal import Decimal

from modelgate.model.data import make_dataset
from modelgate.model.stats import compute_training_stats
from modelgate.serving import metrics
from modelgate.serving.drift import DriftMonitor, psi
from tests.conftest import ADMIN, ARTIFACTS, GOOD_INPUT, assert_reproduces

MANIFEST = json.loads((ARTIFACTS / "manifest.json").read_text())
TRAINING_STATS = MANIFEST["training_stats"]
# The same manifest with every number kept as the decimal the file records, so a comparison can
# hold the regenerated stats to the precision the artifact claims and no further.
RECORDED = json.loads((ARTIFACTS / "manifest.json").read_text(), parse_float=Decimal)


def _rows(n: int, seed: int) -> list[dict]:
    return make_dataset(n, seed)[2]


def _monitor(**overrides) -> DriftMonitor:
    base = dict(window_size=2000, min_samples=100, refresh_every=50)
    base.update(overrides)
    return DriftMonitor(TRAINING_STATS, **base)


def test_psi_is_zero_for_identical_and_grows_with_shift():
    assert psi([0.25] * 4, [0.25] * 4) == 0.0
    mild = psi([0.3, 0.25, 0.25, 0.2], [0.25] * 4)
    severe = psi([0.7, 0.1, 0.1, 0.1], [0.25] * 4)
    assert 0 < mild < 0.1 < severe


def test_manifest_carries_training_stats_for_every_input():
    feats = TRAINING_STATS["features"]
    assert TRAINING_STATS["sample_size"] == 12000
    assert set(feats) == set(GOOD_INPUT)
    assert feats["distance_km"]["kind"] == "numeric" and len(feats["distance_km"]["bin_edges"]) == 9
    assert feats["distance_km"]["bin_edges"] == sorted(feats["distance_km"]["bin_edges"])
    assert feats["pickup_zone_id"]["kind"] == "categorical"
    assert set(feats["pickup_zone_id"]["frequencies"]) == {str(z) for z in range(1, 13)}
    assert abs(feats["is_raining"]["frequencies"]["true"] - 0.2) < 0.02
    # Recomputing from the seeded dataset reproduces every figure the manifest records to within
    # one unit of the last decimal it records. It is not asserted bit for bit: a value on a
    # rounding boundary lands one unit apart between CPU architectures, and the x86-64 CI runner
    # recomputes the distance_km std as 4.990105 against the recorded 4.990104.
    # modelgate/model/stats.py rounds every figure it reports to six decimals, the frequency and
    # quantile tables included, so their category keys inherit their container's tolerance.
    one_place = Decimal("0.000001")
    tolerances = dict.fromkeys(
        ["mean", "std", "min", "max", "bin_edges", "quantiles", "frequencies"], one_place
    )
    regenerated = compute_training_stats(_rows(12000, MANIFEST["seed"]))
    assert_reproduces(regenerated, RECORDED["training_stats"], tolerances, "training_stats")


def test_training_like_traffic_is_stable():
    mon = _monitor()
    for row in _rows(1500, seed=11):
        mon.observe(row)
    rep = mon.report()
    statuses = {name: f["status"] for name, f in rep["features"].items()}
    assert statuses == dict.fromkeys(GOOD_INPUT, "stable"), rep["features"]
    assert rep["drifted"] == []
    for f in rep["features"].values():
        assert f["drift_score"] < 0.1 and f["unknown_rate"] == 0.0
    live = rep["features"]["distance_km"]["live"]
    ref = rep["features"]["distance_km"]["reference"]
    assert abs(live["mean"] - ref["mean"]) < 1.0
    assert abs(live["quantiles"]["p50"] - ref["quantiles"]["p50"]) < 0.5


def test_shifted_distance_drifts_and_other_features_do_not():
    mon = _monitor()
    for row in _rows(1500, seed=12):
        mon.observe({**row, "distance_km": round(row["distance_km"] * 3.0, 3)})
    rep = mon.report()
    dist = rep["features"]["distance_km"]
    assert dist["status"] == "drifted" and dist["drift_score"] > 0.25
    assert dist["live"]["mean"] > dist["reference"]["mean"] * 2.5
    assert rep["drifted"] == ["distance_km"] and rep["worst_feature"] == "distance_km"
    for name, f in rep["features"].items():
        if name != "distance_km":
            assert f["status"] == "stable", (name, f["drift_score"])
    assert metrics.gauge_value(metrics.FEATURE_DRIFT, feature="distance_km") > 0.25
    assert metrics.gauge_value(metrics.FEATURE_DRIFT, feature="traffic_index") < 0.1


def test_categorical_shift_and_unknown_rate():
    mon = _monitor()
    for row in _rows(1200, seed=13):
        mon.observe({**row, "pickup_zone_id": 3, "is_raining": True})
    for _ in range(300):
        mon.record_unknown("pickup_zone_id")
    rep = mon.report()
    zone = rep["features"]["pickup_zone_id"]
    rain = rep["features"]["is_raining"]
    assert zone["status"] == "drifted" and zone["live"]["frequencies"]["3"] == 1.0
    assert rain["status"] == "drifted" and rain["live"]["frequencies"]["true"] == 1.0
    assert zone["unknown_rate"] == 0.2
    assert metrics.gauge_value(metrics.FEATURE_UNKNOWN_RATE, feature="pickup_zone_id") == 0.2
    assert rep["features"]["hour_of_day"]["status"] == "stable"


def test_window_is_bounded_and_recovers_after_shift_passes():
    mon = _monitor(window_size=500, min_samples=100)
    for row in _rows(500, seed=14):
        mon.observe({**row, "traffic_index": 0.99})
    assert mon.report()["features"]["traffic_index"]["status"] == "drifted"
    for row in _rows(500, seed=15):
        mon.observe(row)
    rep = mon.report()
    assert rep["features"]["traffic_index"]["status"] == "stable"
    assert rep["features"]["traffic_index"]["samples"] == 500
    assert metrics.gauge_value(metrics.DRIFT_SAMPLES) == 500


def test_insufficient_samples_report_no_score():
    mon = _monitor(min_samples=100)
    for row in _rows(20, seed=16):
        mon.observe({**row, "distance_km": 400.0})
    rep = mon.report()
    assert rep["features"]["distance_km"]["status"] == "insufficient"
    assert metrics.gauge_value(metrics.FEATURE_DRIFT, feature="distance_km") == 0.0


def test_monitor_without_training_stats_is_disabled():
    mon = DriftMonitor(None)
    mon.observe(GOOD_INPUT)
    assert mon.report() == {"enabled": False, "features": {}}


def test_drift_endpoint_reflects_live_traffic(app, client):
    app.state.drift.reset()
    assert client.get("/admin/drift").status_code == 401
    rep = client.get("/admin/drift", headers=ADMIN).json()
    assert rep["enabled"] and rep["observed"] == 0
    assert rep["features"]["distance_km"]["status"] == "insufficient"

    for row in _rows(300, seed=21):
        assert client.post("/predict", json=row).status_code == 200
    rep = client.get("/admin/drift", headers=ADMIN).json()
    assert rep["observed"] == 300 and rep["drifted"] == []
    assert all(f["status"] == "stable" for f in rep["features"].values())

    for row in _rows(300, seed=22):
        shifted = {**row, "distance_km": round(min(row["distance_km"] * 4.0, 500.0), 3)}
        assert client.post("/predict", json=shifted).status_code == 200
    rep = client.get("/admin/drift", headers=ADMIN).json()
    assert rep["drifted"] == ["distance_km"]
    assert rep["features"]["distance_km"]["drift_score"] > 0.25
    text = client.get("/metrics").text
    line = next(
        ln for ln in text.splitlines() if 'modelgate_feature_drift{feature="distance_km"}' in ln
    )
    assert float(line.split()[-1]) > 0.25

    # Unknown zones are rejected before inference but still count toward the unknown rate.
    for _ in range(150):
        assert client.post("/predict", json={**GOOD_INPUT, "pickup_zone_id": 99}).status_code == 422
    rep = client.get("/admin/drift", headers=ADMIN).json()
    assert rep["features"]["pickup_zone_id"]["unknown_rate"] == 0.2
    assert rep["observed"] == 600

    assert client.post("/admin/drift/reset", headers=ADMIN).json() == {"reset": True}
    rep = client.get("/admin/drift", headers=ADMIN).json()
    assert rep["observed"] == 0 and rep["features"]["pickup_zone_id"]["unknown_rate"] == 0.0
