"""Request and response schemas for the prediction API.

Validation is strict on purpose: unknown fields, wrong types, out-of-range
values, unknown zones, and non-finite floats are all rejected with 422 before
any tensor is built.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, Field
from pydantic_core import PydanticCustomError

from modelgate.model.features import MAX_DISTANCE_KM, ZONE_IDS

_ZONE_SET = frozenset(ZONE_IDS)


def _known_zone(value: int) -> int:
    if value not in _ZONE_SET:
        raise PydanticCustomError(
            "unknown_zone",
            "pickup_zone_id {value} is not a known zone; expected one of {zones}",
            {"value": value, "zones": list(ZONE_IDS)},
        )
    return value


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    distance_km: float = Field(
        ge=0.0, le=MAX_DISTANCE_KM, allow_inf_nan=False, description="Trip distance in km"
    )
    hour_of_day: int = Field(ge=0, le=23, description="Local hour of pickup, 0..23")
    day_of_week: int = Field(ge=0, le=6, description="0 = Monday .. 6 = Sunday")
    pickup_zone_id: Annotated[int, AfterValidator(_known_zone)] = Field(
        description="Pickup zone identifier"
    )
    traffic_index: float = Field(
        ge=0.0, le=1.0, allow_inf_nan=False, description="0 = free flow, 1 = gridlock"
    )
    is_raining: bool = Field(description="Whether it is raining at pickup")


class PredictResponse(BaseModel):
    eta_minutes: float
    model_version: str
    request_id: str


class ShadowRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: str | None = None


class PromoteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: str


class CanaryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    version: str | None = None
    weight: float = Field(default=0.05, gt=0.0, le=1.0, allow_inf_nan=False)


def rejection_reason(error_type: str) -> str:
    """Map a pydantic error type to a low-cardinality rejection reason label."""
    if error_type == "extra_forbidden":
        return "unknown_field"
    if error_type == "missing":
        return "missing_field"
    if error_type == "unknown_zone":
        return "unknown_zone"
    if error_type == "finite_number":
        return "not_finite"
    if error_type in {
        "greater_than",
        "greater_than_equal",
        "less_than",
        "less_than_equal",
    }:
        return "out_of_range"
    if error_type.endswith("_type") or error_type.endswith("_parsing"):
        return "wrong_type"
    if error_type in {"json_invalid", "model_attributes_type"}:
        return "malformed_body"
    return "other"
