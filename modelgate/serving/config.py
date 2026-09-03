"""Runtime settings, read from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    artifacts_dir: Path = Path("artifacts")
    admin_token: str | None = None
    primary_version: str | None = None  # None: lowest version in the manifest
    shadow_version: str | None = None
    shadow_threshold_minutes: float = 2.0
    shadow_log_size: int = 5000
    canary_window_seconds: float = 60.0
    canary_min_samples: int = 50
    canary_error_rate_delta: float = 0.02
    canary_latency_ratio: float = 2.0
    canary_latency_floor_ms: float = 1.0

    @classmethod
    def from_env(cls) -> Settings:
        env = os.environ
        return cls(
            artifacts_dir=Path(env.get("MODELGATE_ARTIFACTS_DIR", "artifacts")),
            admin_token=env.get("MODELGATE_ADMIN_TOKEN") or None,
            primary_version=env.get("MODELGATE_PRIMARY_VERSION") or None,
            shadow_version=env.get("MODELGATE_SHADOW_VERSION") or None,
            shadow_threshold_minutes=float(env.get("MODELGATE_SHADOW_THRESHOLD_MIN", "2.0")),
            shadow_log_size=int(env.get("MODELGATE_SHADOW_LOG_SIZE", "5000")),
            canary_window_seconds=float(env.get("MODELGATE_CANARY_WINDOW_SECONDS", "60")),
            canary_min_samples=int(env.get("MODELGATE_CANARY_MIN_SAMPLES", "50")),
            canary_error_rate_delta=float(env.get("MODELGATE_CANARY_ERROR_RATE_DELTA", "0.02")),
            canary_latency_ratio=float(env.get("MODELGATE_CANARY_LATENCY_RATIO", "2.0")),
            canary_latency_floor_ms=float(env.get("MODELGATE_CANARY_LATENCY_FLOOR_MS", "1.0")),
        )
