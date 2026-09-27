from __future__ import annotations

from decimal import Decimal
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


def _declared_tolerance(path: str, tolerances: dict[str, Decimal]) -> Decimal | None:
    """The tolerance declared for this number, or for the nearest container that holds it."""
    parts = [part.split("[")[0] for part in path.split(".")]
    for part in reversed(parts):
        if part in tolerances:
            return tolerances[part]
    return None


def assert_reproduces(actual, expected, tolerances: dict[str, Decimal], path: str = "") -> None:
    """Assert a regenerated structure matches a committed artifact.

    Structure, strings, booleans and integers must match exactly. A number may differ by the
    tolerance declared for it, because the dataset comes out of torch's vectorised `exp`, whose last
    bit differs between CPU architectures: a value sitting on a rounding boundary rounds one way on
    arm64 and the other on x86-64, and the committed fixture records `distance_km` 8.5 where a Linux
    runner regenerates 8.499.

    `tolerances` maps a key to the largest difference that is not a change in behaviour. For a value
    the generator rounds, that is one unit in its last place; for a value derived from one, it also
    has to cover how far the derivation carries that shift, which the caller states and measures. A
    key may name the number itself or a container whose numbers share a tolerance, which is how a
    frequency table of category keys is covered. A number with no declared tolerance anywhere above
    it must match exactly, so a new field cannot pick one up by accident.
    """
    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"{path}: expected an object, got {type(actual).__name__}"
        assert set(actual) == set(expected), f"{path}: keys differ"
        for key in expected:
            child = f"{path}.{key}" if path else key
            assert_reproduces(actual[key], expected[key], tolerances, child)
    elif isinstance(expected, list):
        assert isinstance(actual, list), f"{path}: expected a list, got {type(actual).__name__}"
        assert len(actual) == len(expected), f"{path}: length {len(actual)} != {len(expected)}"
        for index, item in enumerate(expected):
            assert_reproduces(actual[index], item, tolerances, f"{path}[{index}]")
    elif isinstance(expected, Decimal):
        tolerance = _declared_tolerance(path, tolerances)
        assert tolerance is not None, (
            f"{path}: no declared tolerance, so this number must not be a decimal"
        )
        difference = abs(Decimal(str(actual)) - expected)
        assert difference <= tolerance, (
            f"{path}: {actual} vs {expected} differs by {difference}, over {tolerance}"
        )
    else:
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"


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
