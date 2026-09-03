"""Reference statistics of the training inputs, written to the manifest.

Serving compares live inputs against these to score drift per feature. Numeric
features record mean, std, a few quantiles, and decile edges so the serving
side can bin live values into ten equal-mass bins; categorical features record
the frequency of each category seen in training.
"""

from __future__ import annotations

import math
from typing import Any

NUMERIC_FEATURES: tuple[str, ...] = ("distance_km", "traffic_index")
CATEGORICAL_FEATURES: tuple[str, ...] = (
    "hour_of_day",
    "day_of_week",
    "pickup_zone_id",
    "is_raining",
)
INPUT_FEATURES: tuple[str, ...] = (
    "distance_km",
    "hour_of_day",
    "day_of_week",
    "pickup_zone_id",
    "traffic_index",
    "is_raining",
)
QUANTILES: tuple[int, ...] = (5, 25, 50, 75, 95)
NUM_BINS = 10


def quantile(sorted_values: list[float], pct: float) -> float:
    """Linear-interpolated percentile of an already sorted list."""
    if not sorted_values:
        return 0.0
    pos = (len(sorted_values) - 1) * pct / 100.0
    lo = math.floor(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def mean_std(values: list[float]) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    return mean, math.sqrt(var)


def category_key(value: Any) -> str:
    """Stable JSON-friendly key for a categorical value (bools stay distinct from ints)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def numeric_stats(values: list[float]) -> dict:
    ordered = sorted(float(v) for v in values)
    mean, std = mean_std(ordered)
    edges = [quantile(ordered, 100.0 * i / NUM_BINS) for i in range(1, NUM_BINS)]
    return {
        "kind": "numeric",
        "count": len(ordered),
        "mean": round(mean, 6),
        "std": round(std, 6),
        "min": round(ordered[0], 6) if ordered else 0.0,
        "max": round(ordered[-1], 6) if ordered else 0.0,
        "quantiles": {f"p{p}": round(quantile(ordered, p), 6) for p in QUANTILES},
        "bin_edges": [round(e, 6) for e in edges],
    }


def categorical_stats(values: list[Any]) -> dict:
    counts: dict[str, int] = {}
    for v in values:
        key = category_key(v)
        counts[key] = counts.get(key, 0) + 1
    n = len(values)
    numeric = [float(v) for v in values] if values and not isinstance(values[0], str) else []
    mean, std = mean_std(numeric)
    return {
        "kind": "categorical",
        "count": n,
        "mean": round(mean, 6),
        "std": round(std, 6),
        "frequencies": {k: round(c / n, 6) for k, c in sorted(counts.items(), key=_cat_order)},
    }


def _cat_order(item: tuple[str, int]) -> tuple[int, float | str]:
    key = item[0]
    try:
        return (0, float(key))
    except ValueError:
        return (1, key)


def compute_training_stats(rows: list[dict]) -> dict:
    features: dict[str, dict] = {}
    for name in INPUT_FEATURES:
        column = [row[name] for row in rows]
        if name in NUMERIC_FEATURES:
            features[name] = numeric_stats(column)
        else:
            features[name] = categorical_stats(column)
    return {"sample_size": len(rows), "features": features}
