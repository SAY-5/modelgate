import pytest
import torch
from fastapi.testclient import TestClient

from modelgate.model.features import FEATURE_DIM
from modelgate.serving import metrics
from modelgate.serving.app import create_app
from modelgate.serving.registry import ModelRegistry, NoPrimaryError, UnknownVersionError
from tests.conftest import ADMIN, ARTIFACTS, GOOD_INPUT, make_settings


def test_registry_discovers_versions_from_manifest():
    reg = ModelRegistry(ARTIFACTS)
    assert reg.available_versions() == ["v1", "v2"]
    assert reg.is_known("v1") and not reg.is_known("v9")
    assert not reg.ready


def test_load_warms_and_caches():
    reg = ModelRegistry(ARTIFACTS)
    a = reg.load("v1")
    b = reg.load("v1")
    assert a is b
    assert a.test_mae is not None and a.test_mae > 0
    assert a.predict(torch.zeros(1, FEATURE_DIM)) > 0


def test_unknown_version_raises():
    reg = ModelRegistry(ARTIFACTS)
    with pytest.raises(UnknownVersionError):
        reg.load("v9")


def test_promote_then_rollback():
    reg = ModelRegistry(ARTIFACTS)
    assert reg.primary is None
    with pytest.raises(NoPrimaryError):
        reg.require_primary()

    rec = reg.promote("v1")
    assert rec["changed"] and rec["from"] is None and rec["to"] == "v1"
    assert reg.ready and reg.primary.version == "v1"

    rec = reg.promote("v2")
    assert rec["from"] == "v1" and rec["to"] == "v2"
    assert reg.primary.version == "v2" and reg.previous.version == "v1"

    rec = reg.rollback()
    assert rec["kind"] == "rollback" and reg.primary.version == "v1"
    assert reg.previous.version == "v2"
    assert [s["kind"] for s in reg.swap_history] == ["promote", "promote", "rollback"]


def test_promote_same_version_is_a_noop():
    reg = ModelRegistry(ARTIFACTS)
    reg.promote("v1")
    swaps = metrics.counter_value(metrics.VERSION_SWAPS, kind="promote")
    rec = reg.promote("v1")
    assert rec["changed"] is False
    assert metrics.counter_value(metrics.VERSION_SWAPS, kind="promote") == swaps


def test_rollback_without_history_raises():
    reg = ModelRegistry(ARTIFACTS)
    reg.promote("v1")
    with pytest.raises(NoPrimaryError):
        reg.rollback()


def test_promoting_the_shadow_clears_it():
    reg = ModelRegistry(ARTIFACTS)
    reg.promote("v1")
    reg.set_shadow("v2")
    assert reg.shadow.version == "v2"
    reg.promote("v2")
    assert reg.shadow is None


def test_readyz_false_before_lifespan_then_true():
    app = create_app(make_settings())
    bare = TestClient(app)  # no context manager: lifespan has not run
    r = bare.get("/readyz")
    assert r.status_code == 503 and r.json() == {"ready": False, "primary": None}
    assert bare.get("/healthz").status_code == 200
    with TestClient(app) as ready:
        r = ready.get("/readyz")
        assert r.status_code == 200 and r.json() == {"ready": True, "primary": "v1"}


def test_predict_without_primary_returns_503_and_counts_drop():
    app = create_app(make_settings())
    bare = TestClient(app)
    dropped = metrics.counter_value(metrics.DROPPED_REQUESTS)
    r = bare.post("/predict", json=GOOD_INPUT)
    assert r.status_code == 503
    assert metrics.counter_value(metrics.DROPPED_REQUESTS) == dropped + 1


def test_primary_version_setting_is_honoured():
    app = create_app(make_settings(primary_version="v2", shadow_version="v1"))
    with TestClient(app) as c:
        assert c.get("/readyz").json()["primary"] == "v2"
        assert c.post("/predict", json=GOOD_INPUT).json()["model_version"] == "v2"
        assert c.get("/admin/versions", headers=ADMIN).json()["shadow"] == "v1"


def test_admin_endpoints_require_token(client):
    assert client.get("/admin/versions").status_code == 401
    assert client.get("/admin/versions", headers={"X-Admin-Token": "nope"}).status_code == 401
    assert client.post("/admin/promote", json={"version": "v2"}).status_code == 401
    assert client.post("/admin/rollback").status_code == 401
    assert client.post("/admin/shadow", json={"version": "v2"}).status_code == 401
    assert client.get("/admin/shadow/report").status_code == 401


def test_admin_disabled_without_configured_token():
    app = create_app(make_settings(admin_token=None))
    with TestClient(app) as c:
        assert c.get("/admin/versions", headers=ADMIN).status_code == 503


def test_admin_versions_promote_rollback_flow(client):
    info = client.get("/admin/versions", headers=ADMIN).json()
    assert info["primary"] == "v1" and info["shadow"] is None
    roles = {v["version"]: v["role"] for v in info["versions"]}
    assert roles == {"v1": "primary", "v2": None}

    r = client.post("/admin/promote", json={"version": "v2"}, headers=ADMIN)
    assert r.status_code == 200 and r.json()["to"] == "v2"
    assert client.post("/predict", json=GOOD_INPUT).json()["model_version"] == "v2"

    r = client.post("/admin/rollback", headers=ADMIN)
    assert r.status_code == 200 and r.json()["to"] == "v1"
    assert client.post("/predict", json=GOOD_INPUT).json()["model_version"] == "v1"

    r = client.post("/admin/rollback", headers=ADMIN)
    assert r.status_code == 200 and r.json()["to"] == "v2"


def test_admin_promote_unknown_version_is_404(client):
    r = client.post("/admin/promote", json={"version": "v9"}, headers=ADMIN)
    assert r.status_code == 404
    assert client.post("/admin/shadow", json={"version": "v9"}, headers=ADMIN).status_code == 404


def test_admin_rollback_with_no_history_is_409(client):
    assert client.post("/admin/rollback", headers=ADMIN).status_code == 409
