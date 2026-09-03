"""Feature schema shared by training and serving.

The model consumes a fixed-width float vector. Keeping the encoding in one
place guarantees the serving path builds exactly the tensor the model was
trained on.
"""

from __future__ import annotations

import math

import torch

ZONE_IDS: tuple[int, ...] = tuple(range(1, 13))
NUM_ZONES = len(ZONE_IDS)
_ZONE_INDEX = {zone: i for i, zone in enumerate(ZONE_IDS)}

# Column order of the feature vector. Zone one-hot columns are appended after these.
BASE_FEATURES: tuple[str, ...] = (
    "distance_km",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "traffic_index",
    "is_raining",
)
FEATURE_DIM = len(BASE_FEATURES) + NUM_ZONES

MAX_DISTANCE_KM = 500.0
# Typical trips are a few km; scaling by 50 keeps the dominant feature in a
# range the MLP learns from quickly while still covering the 500 km cap.
DISTANCE_SCALE_KM = 50.0


def feature_names() -> list[str]:
    return list(BASE_FEATURES) + [f"zone_{z}" for z in ZONE_IDS]


def encode(
    distance_km: float,
    hour_of_day: int,
    day_of_week: int,
    pickup_zone_id: int,
    traffic_index: float,
    is_raining: bool,
) -> list[float]:
    """Encode one trip into the model's flat feature vector."""
    hour_angle = 2.0 * math.pi * hour_of_day / 24.0
    dow_angle = 2.0 * math.pi * day_of_week / 7.0
    row = [
        distance_km / DISTANCE_SCALE_KM,
        math.sin(hour_angle),
        math.cos(hour_angle),
        math.sin(dow_angle),
        math.cos(dow_angle),
        traffic_index,
        1.0 if is_raining else 0.0,
    ]
    zone = [0.0] * NUM_ZONES
    zone[_ZONE_INDEX[pickup_zone_id]] = 1.0
    return row + zone


def encode_batch(rows: list[dict]) -> torch.Tensor:
    return torch.tensor(
        [
            encode(
                r["distance_km"],
                r["hour_of_day"],
                r["day_of_week"],
                r["pickup_zone_id"],
                r["traffic_index"],
                r["is_raining"],
            )
            for r in rows
        ],
        dtype=torch.float32,
    )
