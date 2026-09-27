"""Measure how far the fixture's eta fields move when its inputs land one unit apart.

`distance_km` and `traffic_index` are rounded to 3 and 4 decimals, so a value on a rounding
boundary can come out one unit apart on two hosts. This moves both inputs of every record in
tests/fixtures/replay_log.jsonl by one unit in their last place, in every sign combination, and
prints the largest change in the reference eta (`true_eta_minutes`, which `actual_eta_minutes`
adds its noise to) and in the prediction v1 serves (`eta_minutes`), rounded up to four decimals
as the FIXTURE_TOLERANCES comment in tests/test_eval.py quotes them.

    uv run python tests/measure_eta_shift.py
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import torch

from modelgate.model.data import true_eta_minutes
from modelgate.model.features import encode
from modelgate.serving.registry import ModelRegistry

TESTS = Path(__file__).resolve().parent
FIXTURE = TESTS / "fixtures" / "replay_log.jsonl"
ARTIFACTS = TESTS.parent / "artifacts"
# One unit in the last place modelgate/model/data.py rounds each input to, and its decimals.
UNITS = {"distance_km": (0.001, 3), "traffic_index": (0.0001, 4)}


def neighbours(row: dict) -> list[dict]:
    """The row with every input moved one unit up or down, in every combination."""
    moved = []
    for signs in itertools.product((-1, 1), repeat=len(UNITS)):
        shifted = dict(row)
        for sign, (name, (unit, places)) in zip(signs, UNITS.items(), strict=True):
            shifted[name] = round(row[name] + sign * unit, places)
        moved.append(shifted)
    return moved


def main() -> None:
    v1 = ModelRegistry(ARTIFACTS).load("v1")

    def served(rows: list[dict]) -> list[float]:
        return v1.predict_batch(torch.tensor([encode(**r) for r in rows], dtype=torch.float32))

    rows = [json.loads(line)["input"] for line in FIXTURE.read_text().splitlines()]
    reference = prediction = 0.0
    for row, served_eta in zip(rows, served(rows), strict=True):
        moved = neighbours(row)
        base = true_eta_minutes(**row)
        reference = max(reference, *(abs(true_eta_minutes(**m) - base) for m in moved))
        prediction = max(prediction, *(abs(p - served_eta) for p in served(moved)))
    for label, shift in (("reference eta", reference), ("served prediction", prediction)):
        bound = math.ceil(shift * 10_000) / 10_000
        print(f"{label}: at most {bound:.4f} minutes (largest shift {shift:.6f})")


if __name__ == "__main__":
    main()
