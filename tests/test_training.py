import json

import torch

from modelgate.model import train
from modelgate.model.data import make_dataset, true_eta_minutes
from modelgate.model.features import FEATURE_DIM, ZONE_IDS, encode, feature_names
from modelgate.model.net import load_model


def test_feature_encoding_shape_and_cyclic_hours():
    row = encode(10.0, 0, 0, ZONE_IDS[0], 0.5, False)
    assert len(row) == FEATURE_DIM == len(feature_names())
    assert row[0] == 10.0 / 50.0
    assert row[7] == 1.0 and sum(row[7:]) == 1.0
    midnight = encode(1, 0, 0, 1, 0, False)[1:3]
    almost_midnight = encode(1, 23, 0, 1, 0, False)[1:3]
    noon = encode(1, 12, 0, 1, 0, False)[1:3]
    assert abs(midnight[0] - almost_midnight[0]) < 0.3  # adjacent hours are close
    assert abs(midnight[1] - noon[1]) > 1.9  # opposite hours are far apart


def test_dataset_is_deterministic():
    x1, y1, rows1 = make_dataset(200, seed=3)
    x2, y2, rows2 = make_dataset(200, seed=3)
    x3, _, _ = make_dataset(200, seed=4)
    assert torch.equal(x1, x2) and torch.equal(y1, y2) and rows1 == rows2
    assert not torch.equal(x1, x3)
    assert x1.shape == (200, FEATURE_DIM) and (y1 >= 1.0).all()


def test_true_eta_is_monotone_in_distance_and_traffic():
    base = dict(
        hour_of_day=10, day_of_week=1, pickup_zone_id=1, traffic_index=0.3, is_raining=False
    )
    assert true_eta_minutes(5, **base) < true_eta_minutes(10, **base)
    assert true_eta_minutes(5, **{**base, "traffic_index": 0.9}) > true_eta_minutes(5, **base)
    assert true_eta_minutes(5, **{**base, "is_raining": True}) > true_eta_minutes(5, **base)


def test_training_is_reproducible_and_beats_baselines(tmp_path):
    m1 = train.train_all(tmp_path / "a", seed=7, quiet=True)
    m2 = train.train_all(tmp_path / "b", seed=7, quiet=True)

    for version in ("v1", "v2"):
        mae1 = m1["versions"][version]["metrics"]["test_mae_minutes"]
        mae2 = m2["versions"][version]["metrics"]["test_mae_minutes"]
        assert abs(mae1 - mae2) < 1e-3, (version, mae1, mae2)
        assert mae1 < m1["baselines"]["mean_predictor_mae"]
        assert mae1 < m1["baselines"]["distance_linear_mae"]
        a = load_model(tmp_path / "a" / f"eta_{version}.pt").state_dict()
        b = load_model(tmp_path / "b" / f"eta_{version}.pt").state_dict()
        assert all(torch.allclose(a[k], b[k]) for k in a)

    assert (
        m1["versions"]["v2"]["metrics"]["test_mae_minutes"]
        < (m1["versions"]["v1"]["metrics"]["test_mae_minutes"])
    )
    manifest = json.loads((tmp_path / "a" / "manifest.json").read_text())
    assert manifest["feature_schema"]["dim"] == FEATURE_DIM
    assert set(manifest["versions"]) == {"v1", "v2"}


def test_committed_artifacts_match_manifest_schema():
    from tests.conftest import ARTIFACTS

    manifest = json.loads((ARTIFACTS / "manifest.json").read_text())
    assert manifest["feature_schema"]["names"] == feature_names()
    for version, entry in manifest["versions"].items():
        net = load_model(ARTIFACTS / entry["file"])
        out = net(torch.zeros(2, FEATURE_DIM))
        assert out.shape == (2,) and (out > 0).all(), version
