from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from modelgate.serving.app import create_app
from modelgate.serving.config import Settings

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"
ADMIN_TOKEN = "test-token"
ADMIN = {"X-Admin-Token": ADMIN_TOKEN}

GOOD_INPUT = {
    "distance_km": 8.2,
    "hour_of_day": 17,
    "day_of_week": 2,
    "pickup_zone_id": 3,
    "traffic_index": 0.7,
    "is_raining": True,
}


def make_settings(**overrides) -> Settings:
    base = {"artifacts_dir": ARTIFACTS, "admin_token": ADMIN_TOKEN}
    base.update(overrides)
    return Settings(**base)


@pytest.fixture
def app():
    return create_app(make_settings())


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c
