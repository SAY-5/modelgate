"""What `assert_reproduces` admits when a committed artifact is regenerated.

A committed artifact does not regenerate bit for bit on a different CPU architecture: a value
sitting on a rounding boundary lands one unit apart between the two hosts, and the fixture, made
on arm64, records `distance_km` 8.499 where the x86-64 CI runner regenerates 8.5. The comparison
therefore admits the tolerance declared for each field, one unit in the last place for a value
the generator rounds and a stated allowance for a value derived from one, and nothing looser.
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
# The tolerances the fixture test declares. 8.5 is written with one decimal because JSON drops the
# trailing zeros, which is exactly why a tolerance cannot be read off the text of a file.
TOLERANCES = {
    "distance_km": Decimal("0.001"),
    "traffic_index": Decimal("0.0001"),
    "eta_minutes": Decimal("0.02"),
}
REGENERATED = {
    "at": 1,
    "input": {"distance_km": 8.5, "traffic_index": 0.7299},
    "eta_minutes": 23.17,
    "is_raining": True,
}


def _with_input(**changes: object) -> dict:
    return {**REGENERATED, "input": {**REGENERATED["input"], **changes}}


def test_identical_records_match():
    assert_reproduces(dict(REGENERATED), RECORDED, TOLERANCES)


def test_a_difference_within_the_declared_tolerance_is_admitted():
    # The difference this whole tolerance exists for, in both directions, and on a derived field
    # whose tolerance carries an input shift through the model.
    assert_reproduces(_with_input(distance_km=8.499), RECORDED, TOLERANCES)
    assert_reproduces(_with_input(traffic_index=0.7298), RECORDED, TOLERANCES)
    assert_reproduces(_with_input(traffic_index=0.73), RECORDED, TOLERANCES)
    assert_reproduces({**REGENERATED, "eta_minutes": 23.18}, RECORDED, TOLERANCES)
    assert_reproduces({**REGENERATED, "eta_minutes": 23.19}, RECORDED, TOLERANCES)


@pytest.mark.parametrize(
    "regenerated",
    [
        _with_input(distance_km=8.498),
        _with_input(traffic_index=0.8299),
        {**REGENERATED, "eta_minutes": 23.67},
        {**REGENERATED, "eta_minutes": 23.2},
        {**REGENERATED, "at": 2},
        {**REGENERATED, "is_raining": False},
        {key: value for key, value in REGENERATED.items() if key != "eta_minutes"},
        {**REGENERATED, "input": [8.5, 0.7299]},
    ],
)
def test_anything_larger_or_structural_fails(regenerated):
    with pytest.raises(AssertionError):
        assert_reproduces(regenerated, RECORDED, TOLERANCES)


def test_a_decimal_without_a_declared_tolerance_is_refused_even_when_equal():
    recorded = json.loads('{"unmapped":1.25}', parse_float=Decimal)
    assert_reproduces({"unmapped": 1.25}, recorded, {"unmapped": Decimal("0.01")})
    with pytest.raises(AssertionError, match="no declared tolerance"):
        assert_reproduces({"unmapped": 1.25}, recorded, TOLERANCES)


def test_the_failure_names_the_field_and_the_tolerance():
    with pytest.raises(
        AssertionError, match=r"input\.distance_km: 8\.498 vs 8\.5 differs by 0\.002, over 0\.001"
    ):
        assert_reproduces(_with_input(distance_km=8.498), RECORDED, TOLERANCES)
