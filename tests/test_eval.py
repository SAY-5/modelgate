"""Request log capture and `modelgate eval` replay: determinism and report math."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import torch
from fastapi.testclient import TestClient

from modelgate.cli import main as cli_main
from modelgate.eval import (
    accuracy_report,
    divergence_report,
    evaluate,
    load_log,
    logged_consistency,
    replay,
    truth_values,
)
from modelgate.model.data import make_dataset
from modelgate.model.features import encode
from modelgate.serving import metrics
from modelgate.serving.app import create_app
from modelgate.serving.registry import ModelRegistry
from modelgate.serving.reqlog import INPUT_FIELDS, RequestLog
from tests.conftest import ADMIN, ARTIFACTS, GOOD_INPUT, assert_reproduces, make_settings

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "replay_log.jsonl"
FIXTURE_SIZE = 300
# The largest difference in each field that is not a change in behaviour. The two inputs are one
# unit in the last place the generator rounds them to. The two eta fields are derived from those
# inputs, so their tolerance also has to carry that shift through: measured over this fixture by
# tests/measure_eta_shift.py, one unit in the last place of each input moves the reference eta by
# at most 0.0094 minutes and the served prediction by at most 0.0028, so 0.02 covers the shift and
# the field's own rounding with room to spare, while still being far tighter than any real change
# in the model or the formula.
FIXTURE_TOLERANCES = {
    "distance_km": Decimal("0.001"),
    "traffic_index": Decimal("0.0001"),
    "eta_minutes": Decimal("0.02"),
    "actual_eta_minutes": Decimal("0.02"),
}
FIXTURE_SEED = 99


def make_fixture_records() -> list[dict]:
    """300 seeded trips served by v1, with the noisy synthetic actual attached."""
    x, y, rows = make_dataset(FIXTURE_SIZE, FIXTURE_SEED)
    v1 = ModelRegistry(ARTIFACTS).load("v1")
    etas = v1.predict_batch(x)
    return [
        {
            "at": 1_756_900_000 + i,
            "input": row,
            "version": "v1",
            "eta_minutes": round(eta, 2),
            "actual_eta_minutes": round(float(y[i]), 3),
        }
        for i, (row, eta) in enumerate(zip(rows, etas, strict=True))
    ]


def test_fixture_is_reproducible_from_the_seeded_dataset():
    # Record for record rather than byte for byte: a value on a rounding boundary lands one unit
    # apart between CPU architectures, and the committed fixture, made on arm64, records distance_km
    # 8.499 where the x86-64 CI runner regenerates 8.5. Each number is therefore held to the
    # tolerance FIXTURE_TOLERANCES declares for its field, and the keys, the order, the line count
    # and every integer, string and boolean still have to match.
    recorded = [json.loads(line, parse_float=Decimal) for line in FIXTURE.read_text().splitlines()]
    regenerated = make_fixture_records()
    assert len(recorded) == len(regenerated) == FIXTURE_SIZE
    for i, (actual, expected) in enumerate(zip(regenerated, recorded, strict=True)):
        assert list(actual) == list(expected), f"record {i}: field order differs"
        assert_reproduces(actual, expected, FIXTURE_TOLERANCES, f"record {i}")
    # The file is still one compact JSON object per line with a trailing newline.
    assert FIXTURE.read_text().endswith("}\n")
    assert all(
        line.startswith('{"at":') and ", " not in line for line in FIXTURE.read_text().splitlines()
    )


def test_load_log_validates_inputs_and_counts_skips(tmp_path):
    good = make_fixture_records()[:3]
    bad_input = {**good[0], "input": {**good[0]["input"], "pickup_zone_id": 99}}
    lines = [json.dumps(r) for r in good] + [json.dumps(bad_input), "not json", "{}", ""]
    path = tmp_path / "log.jsonl"
    path.write_text("\n".join(lines) + "\n")
    records, skipped = load_log(path)
    assert len(records) == 3
    assert skipped == {"malformed": 2, "invalid_input": 1}
    assert records[0]["input"] == good[0]["input"]


def test_truth_modes():
    records = make_fixture_records()[:5]
    assert truth_values(records, "none") == [None] * 5
    ref = truth_values(records, "reference")
    auto = truth_values(records, "auto")
    assert all(isinstance(v, float) for v in ref) and all(isinstance(v, float) for v in auto)
    assert auto == [r["actual_eta_minutes"] for r in records]
    stripped = [{k: v for k, v in r.items() if k != "actual_eta_minutes"} for r in records]
    assert truth_values(stripped, "auto") == [None] * 5


def test_accuracy_report_math_by_hand():
    preds = [4.0, 12.0, 12.0, 50.0]
    truths = [5.0, 10.0, 15.0, None]
    acc = accuracy_report(preds, truths)
    assert acc["n"] == 3
    assert acc["mae_minutes"] == round((1 + 2 + 3) / 3, 4)
    assert acc["bias_minutes"] == round((-1 + 2 - 3) / 3, 4)
    assert acc["share_within_2min"] == round(2 / 3, 4)
    buckets = {b["bucket"]: b for b in acc["calibration"]["buckets"]}
    assert set(buckets) == {"0-5", "10-15"}
    assert buckets["0-5"] == {
        "bucket": "0-5",
        "n": 1,
        "mean_predicted": 4.0,
        "mean_actual": 5.0,
        "gap": -1.0,
    }
    assert buckets["10-15"]["n"] == 2 and buckets["10-15"]["mean_actual"] == 12.5
    assert buckets["10-15"]["gap"] == -0.5
    # ECE is the sample-weighted mean absolute bucket gap: (1*1 + 2*0.5) / 3.
    assert acc["calibration"]["expected_calibration_error"] == round(2 / 3, 4)
    assert accuracy_report([1.0], [None])["n"] == 0


def test_divergence_report_matches_shadow_math():
    a = [10.0, 20.0, 30.0, 40.0]
    b = [10.5, 23.0, 29.0, 46.0]
    div = divergence_report(a, b, threshold=2.0)
    assert div["n"] == 4
    assert div["abs_delta_minutes"] == {"mean": 2.625, "p50": 1.0, "p95": 6.0, "max": 6.0}
    assert div["share_beyond_threshold"] == 0.5
    assert div["bias_minutes"] == 2.125
    assert div["agreement_within_1min"] == 0.5
    assert divergence_report([], []) == {"n": 0}


def test_logged_consistency_flags_only_real_gaps():
    records = [
        {"version": "v1", "eta_minutes": 10.0},
        {"version": "v1", "eta_minutes": 10.5},
        {"version": "v2", "eta_minutes": 99.0},
        {"version": "v1"},
    ]
    preds = {"v1": [10.004, 10.0, 1.0, 1.0]}
    con = logged_consistency(records, preds)
    assert con == {"checked": 2, "mismatches": 1, "max_gap_minutes": 0.5}


def test_replay_is_deterministic_and_matches_serving_answers():
    registry = ModelRegistry(ARTIFACTS)
    records, skipped = load_log(FIXTURE)
    assert len(records) == FIXTURE_SIZE and skipped == {"malformed": 0, "invalid_input": 0}
    first = replay(registry, ["v1", "v2"], records)
    second = replay(ModelRegistry(ARTIFACTS), ["v2", "v1"], records)
    assert first["v1"] == second["v1"] and first["v2"] == second["v2"]
    # The fixture's logged ETAs were produced by the same padded path the service uses.
    assert [round(e, 2) for e in first["v1"]] == [r["eta_minutes"] for r in records]
    single = registry.load("v2").predict(torch.tensor([encode(**records[7]["input"])]))
    assert first["v2"][7] == single


def test_end_to_end_report_on_fixture():
    report = evaluate(FIXTURE, ["v1", "v2"], ARTIFACTS, truth="auto")
    assert report["records"] == FIXTURE_SIZE and report["truth_available"] == FIXTURE_SIZE
    v1, v2 = report["versions"]["v1"], report["versions"]["v2"]
    assert v1["n"] == v2["n"] == FIXTURE_SIZE
    # Manifest MAEs are 3.1 (v1) and 1.9 (v2) on the held-out split; this sample agrees.
    assert 2.0 < v1["mae_minutes"] < 4.5
    assert 1.0 < v2["mae_minutes"] < 3.0
    assert v2["mae_minutes"] < v1["mae_minutes"]
    assert sum(b["n"] for b in v1["calibration"]["buckets"]) == FIXTURE_SIZE
    assert 0 <= v2["calibration"]["expected_calibration_error"] < 3
    assert report["consistency"] == {"checked": FIXTURE_SIZE, "mismatches": 0, "max_gap_minutes": 0}
    div = report["divergence"]
    assert div["from"] == "v1" and div["to"] == "v2" and div["n"] == FIXTURE_SIZE
    assert 0 < div["abs_delta_minutes"]["mean"] < 10
    again = evaluate(FIXTURE, ["v1", "v2"], ARTIFACTS, truth="auto")
    for key in ("versions", "consistency", "divergence"):
        assert again[key] == report[key]
    ref = evaluate(FIXTURE, ["v2"], ARTIFACTS, truth="reference")
    assert ref["divergence"] is None and ref["versions"]["v2"]["mae_minutes"] < 3.0
    none = evaluate(FIXTURE, ["v1"], ARTIFACTS, truth="none")
    assert none["versions"]["v1"]["n"] == 0 and none["truth_available"] == 0


def test_cli_eval_prints_report_and_writes_json(tmp_path, capsys):
    out = tmp_path / "report.json"
    code = cli_main(
        [
            "eval",
            "--log",
            str(FIXTURE),
            "--versions",
            "v1",
            "v2",
            "--artifacts",
            str(ARTIFACTS),
            "--json",
            str(out),
        ]
    )
    assert code == 0
    text = capsys.readouterr().out
    assert "ModelGate replay evaluation" in text
    assert "v1     n=300  MAE" in text and "v2     n=300  MAE" in text
    assert "divergence v1 -> v2" in text and "logged vs replayed: checked 300, mismatches 0" in text
    written = json.loads(out.read_text())
    assert written["records"] == 300 and written["log"] == str(FIXTURE)
    assert set(written["versions"]) == {"v1", "v2"}


def test_request_log_samples_exactly_and_stays_pii_free(tmp_path):
    path = tmp_path / "requests.jsonl"
    written = metrics.counter_value(metrics.REQUEST_LOG_RECORDS)
    app = create_app(make_settings(request_log_path=path, request_log_sample_rate=0.5))
    rows = make_dataset(100, seed=41)[2]
    with TestClient(app) as client:
        for row in rows:
            r = client.post("/predict", json=row, headers={"X-Request-Id": "user-42@example.com"})
            assert r.status_code == 200
        assert client.post("/predict", json={**GOOD_INPUT, "hour_of_day": 30}).status_code == 422
        status = client.get("/admin/request-log", headers=ADMIN).json()
        assert status == {"enabled": True, "path": str(path), "sample_rate": 0.5, "records": 50}
    assert metrics.counter_value(metrics.REQUEST_LOG_RECORDS) == written + 50

    lines = path.read_text().splitlines()
    assert len(lines) == 50
    for line in lines:
        rec = json.loads(line)
        assert set(rec) == {"at", "input", "version", "eta_minutes"}
        assert set(rec["input"]) == set(INPUT_FIELDS)
        assert rec["version"] == "v1"
        assert "user-42" not in line and "example.com" not in line
    # Every second accepted request was written, in order.
    assert [json.loads(ln)["input"] for ln in lines] == rows[1::2]

    # Replaying the captured log reproduces the served answers exactly.
    report = evaluate(path, ["v1", "v2"], ARTIFACTS, truth="none")
    assert report["records"] == 50
    assert report["consistency"] == {"checked": 50, "mismatches": 0, "max_gap_minutes": 0}


def test_request_log_disabled_without_path_or_with_zero_rate(tmp_path):
    off = RequestLog(None, 0.5)
    assert off.record(GOOD_INPUT, "v1", 1.0) is False and off.describe()["enabled"] is False
    zero = RequestLog(tmp_path / "zero.jsonl", 0.0)
    assert zero.record(GOOD_INPUT, "v1", 1.0) is False
    app = create_app(make_settings())
    with TestClient(app) as client:
        client.post("/predict", json=GOOD_INPUT)
        status = client.get("/admin/request-log", headers=ADMIN).json()
        assert status["enabled"] is False and status["records"] == 0
