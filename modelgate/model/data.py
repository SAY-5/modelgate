"""Synthetic trip dataset with a fixed seed.

The ETA formula is intentionally simple but non-linear: base speed drops with
traffic and rain, night trips are faster, and each pickup zone carries an
offset for its access time. Noise is added so the MLP has something real to
fit rather than a lookup table.
"""

from __future__ import annotations

import torch

from modelgate.model.features import NUM_ZONES, ZONE_IDS, encode

ZONE_OFFSET_MIN = [2.0, 1.0, 4.5, 3.0, 0.5, 6.0, 2.5, 1.5, 5.0, 3.5, 0.0, 4.0]


def true_eta_minutes(
    distance_km: float,
    hour_of_day: int,
    day_of_week: int,
    pickup_zone_id: int,
    traffic_index: float,
    is_raining: bool,
) -> float:
    """Reference formula the synthetic dataset is built from."""
    base_speed = 38.0  # km/h on an empty road
    speed = base_speed * (1.0 - 0.55 * traffic_index)
    if is_raining:
        speed *= 0.82
    if hour_of_day < 6 or hour_of_day >= 22:
        speed *= 1.15
    if day_of_week >= 5:
        speed *= 1.08
    travel = 60.0 * distance_km / max(speed, 4.0)
    pickup = ZONE_OFFSET_MIN[ZONE_IDS.index(pickup_zone_id)]
    return travel + pickup


def make_dataset(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    """Build n trips deterministically from seed. Returns (X, y, raw_rows)."""
    g = torch.Generator().manual_seed(seed)
    dist = torch.exp(torch.randn(n, generator=g) * 0.7 + 1.6).clamp(0.3, 120.0)
    hour = torch.randint(0, 24, (n,), generator=g)
    dow = torch.randint(0, 7, (n,), generator=g)
    zone_idx = torch.randint(0, NUM_ZONES, (n,), generator=g)
    # Traffic peaks at commute hours; beta-ish shape via clamped normal.
    peak = ((hour - 8).abs().float().clamp(max=9) / 9.0).minimum(
        (hour - 17).abs().float().clamp(max=9) / 9.0
    )
    traffic = (0.75 - 0.5 * peak + 0.18 * torch.randn(n, generator=g)).clamp(0.0, 1.0)
    rain = torch.rand(n, generator=g) < 0.2
    noise = torch.randn(n, generator=g) * 1.8

    rows: list[dict] = []
    xs: list[list[float]] = []
    ys: list[float] = []
    for i in range(n):
        row = {
            "distance_km": round(float(dist[i]), 3),
            "hour_of_day": int(hour[i]),
            "day_of_week": int(dow[i]),
            "pickup_zone_id": ZONE_IDS[int(zone_idx[i])],
            "traffic_index": round(float(traffic[i]), 4),
            "is_raining": bool(rain[i]),
        }
        eta = true_eta_minutes(**row) + float(noise[i])
        eta = max(eta, 1.0)
        rows.append(row)
        xs.append(encode(**row))
        ys.append(eta)
    x = torch.tensor(xs, dtype=torch.float32)
    y = torch.tensor(ys, dtype=torch.float32)
    return x, y, rows


def mean_absolute_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float((pred - target).abs().mean())


def naive_baseline_mae(y_train: torch.Tensor, y_test: torch.Tensor) -> float:
    """MAE of always predicting the training-set mean."""
    return mean_absolute_error(torch.full_like(y_test, float(y_train.mean())), y_test)


def distance_baseline_mae(
    x_train: torch.Tensor, y_train: torch.Tensor, x_test: torch.Tensor, y_test: torch.Tensor
) -> float:
    """MAE of a one-variable linear fit on distance (a stronger but still naive baseline)."""
    d_train = x_train[:, 0]
    d_test = x_test[:, 0]
    d_mean = d_train.mean()
    y_mean = y_train.mean()
    slope = ((d_train - d_mean) * (y_train - y_mean)).sum() / ((d_train - d_mean) ** 2).sum()
    intercept = y_mean - slope * d_mean
    return mean_absolute_error(slope * d_test + intercept, y_test)


__all__ = [
    "make_dataset",
    "true_eta_minutes",
    "mean_absolute_error",
    "naive_baseline_mae",
    "distance_baseline_mae",
]
