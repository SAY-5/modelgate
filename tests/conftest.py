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


def _declared_places(path: str, decimals: dict[str, int]) -> int | None:
    """The precision declared for this number, or for the nearest container that holds it."""
    parts = [part.split("[")[0] for part in path.split(".")]
    for part in reversed(parts):
        if part in decimals:
            return decimals[part]
    return None


def assert_reproduces(actual, expected, decimals: dict[str, int], path: str = "") -> None:
    """Assert a regenerated structure matches a committed artifact.

    Structure, strings, booleans and integers must match exactly. A number may differ by one unit
    in the last place the code that produced it rounds to, because the dataset comes out of torch's
    vectorised `exp`, whose last bit differs between CPU architectures: a value sitting on a
    rounding boundary rounds one way on arm64 and the other on x86-64, and the committed fixture
    records `distance_km` 8.5 where a Linux runner regenerates 8.499.

    `decimals` maps a key to the number of decimals its producer rounds to, so the tolerance comes
    from that code rather than from the text of the file, which drops trailing zeros and would make
    8.5 look like a one decimal field. The key may name the number itself or a container whose
    numbers all share a precision, which is how a frequency table of category keys is covered. A
    number with no declared precision anywhere above it must match exactly, so a new field cannot
    pick up a tolerance by accident.
    """
    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"{path}: expected an object, got {type(actual).__name__}"
        assert set(actual) == set(expected), f"{path}: keys differ"
        for key in expected:
            child = f"{path}.{key}" if path else key
            assert_reproduces(actual[key], expected[key], decimals, child)
    elif isinstance(expected, list):
        assert isinstance(actual, list), f"{path}: expected a list, got {type(actual).__name__}"
        assert len(actual) == len(expected), f"{path}: length {len(actual)} != {len(expected)}"
        for index, item in enumerate(expected):
            assert_reproduces(actual[index], item, decimals, f"{path}[{index}]")
    elif isinstance(expected, Decimal):
        places = _declared_places(path, decimals)
        assert places is not None, (
            f"{path}: no declared precision, so this number must not be a decimal"
        )
        tolerance = Decimal(1).scaleb(-places)
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
