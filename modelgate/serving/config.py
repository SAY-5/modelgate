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
        )
