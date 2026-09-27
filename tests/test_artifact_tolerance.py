"""What `assert_reproduces` admits when a committed artifact is regenerated.

The committed dataset comes out of torch's vectorised `exp`, whose last bit differs between CPU
architectures, so a value sitting on a rounding boundary rounds one way on arm64 and the other on
x86-64: the fixture records `distance_km` 8.5 where a Linux runner regenerates 8.499. The
comparison therefore admits one unit in the last decimal a file records, and nothing looser.
"""

from __future__ import annotations

import json
from decimal import Decimal

import pytest

from tests.conftest import assert_reproduces

RECORDED = json.loads(
    '{"at":1,"input":{"distance_km":8.5,"traffic_index":0.7299},"eta_minutes":23.17,"is_raining":true}',
    parse_float=Decimal,
)
# What the producers round to, the same values the fixture test declares. 8.5 is written with one
# decimal because JSON drops the trailing zeros, which is exactly why the tolerance cannot be read
# off the text.
DECIMALS = {"distance_km": 3, "traffic_index": 4, "eta_minutes": 2}
REGENERATED = {
    "at": 1,
    "input": {"distance_km": 8.5, "traffic_index": 0.7299},
    "eta_minutes": 23.17,
    "is_raining": True,
}


def _with_input(**changes: object) -> dict:
    return {**REGENERATED, "input": {**REGENERATED["input"], **changes}}


def test_identical_records_match():
    assert_reproduces(dict(REGENERATED), RECORDED, DECIMALS)


def test_one_unit_in_the_last_recorded_place_is_admitted():
    # The difference this whole tolerance exists for, and the same size on a four decimal field.
    assert_reproduces(_with_input(distance_km=8.499), RECORDED, DECIMALS)
    assert_reproduces(_with_input(traffic_index=0.7298), RECORDED, DECIMALS)
    assert_reproduces(_with_input(traffic_index=0.73), RECORDED, DECIMALS)
    assert_reproduces({**REGENERATED, "eta_minutes": 23.18}, RECORDED, DECIMALS)


@pytest.mark.parametrize(
    "regenerated",
    [
        _with_input(distance_km=8.498),
        _with_input(traffic_index=0.8299),
        {**REGENERATED, "eta_minutes": 23.67},
        {**REGENERATED, "at": 2},
        {**REGENERATED, "is_raining": False},
        {key: value for key, value in REGENERATED.items() if key != "eta_minutes"},
        {**REGENERATED, "input": [8.5, 0.7299]},
    ],
)
def test_anything_larger_or_structural_fails(regenerated):
    with pytest.raises(AssertionError):
        assert_reproduces(regenerated, RECORDED, DECIMALS)


def test_a_decimal_without_a_declared_precision_must_match_exactly():
    recorded = json.loads('{"unmapped":1.25}', parse_float=Decimal)
    assert_reproduces({"unmapped": 1.25}, recorded, {"unmapped": 2})
    with pytest.raises(AssertionError, match="no declared precision"):
        assert_reproduces({"unmapped": 1.25}, recorded, DECIMALS)


def test_the_failure_names_the_field_and_the_tolerance():
    with pytest.raises(
        AssertionError, match=r"input\.distance_km: 8\.498 vs 8\.5 differs by 0\.002, over 0\.001"
    ):
        assert_reproduces(_with_input(distance_km=8.498), RECORDED, DECIMALS)
