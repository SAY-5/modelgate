"""Model registry with atomic, zero-drop version swaps.

A version is loaded from disk and warmed with a dummy inference off the
request path. Only after that succeeds is the primary reference replaced, and
the replacement is a single attribute assignment taken under a lock. Every
request captures its own reference to the primary at the start, so a request
in flight during a swap finishes on the model it started with. No request
observes a partially loaded model.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from modelgate.model.features import FEATURE_DIM
from modelgate.model.net import EtaNet, load_model
from modelgate.serving import metrics


class UnknownVersionError(KeyError):
    pass


class NoPrimaryError(RuntimeError):
    pass


@dataclass(frozen=True)
class LoadedModel:
    version: str
    net: EtaNet
    test_mae: float | None
    loaded_at: float = field(default_factory=time.time)

    def predict(self, features: torch.Tensor) -> float:
        with torch.inference_mode():
            return float(self.net(features)[0])


class ModelRegistry:
    def __init__(self, artifacts_dir: Path) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self._manifest = self._read_manifest()
        self._loaded: dict[str, LoadedModel] = {}
        self._primary: LoadedModel | None = None
        self._previous: LoadedModel | None = None
        self._shadow: LoadedModel | None = None
        self._swap_lock = threading.Lock()
        self._load_lock = threading.Lock()
        self.swap_history: list[dict] = []

    # ---- discovery -------------------------------------------------------

    def _read_manifest(self) -> dict:
        path = self.artifacts_dir / "manifest.json"
        if not path.exists():
            return {"versions": {}}
        return json.loads(path.read_text())

    def available_versions(self) -> list[str]:
        return sorted(self._manifest.get("versions", {}))

    def is_known(self, version: str) -> bool:
        return version in self._manifest.get("versions", {})

    # ---- loading ---------------------------------------------------------

    def load(self, version: str) -> LoadedModel:
        """Load and warm a version. Idempotent; safe to call from any thread."""
        if not self.is_known(version):
            raise UnknownVersionError(version)
        cached = self._loaded.get(version)
        if cached is not None:
            return cached
        with self._load_lock:
            cached = self._loaded.get(version)
            if cached is not None:
                return cached
            entry = self._manifest["versions"][version]
            net = load_model(self.artifacts_dir / entry["file"])
            net.eval()
            loaded = LoadedModel(
                version=version,
                net=net,
                test_mae=entry.get("metrics", {}).get("test_mae_minutes"),
            )
            # Warm-up: first forward pass pays for lazy kernel init, so take it here.
            loaded.predict(torch.zeros(1, FEATURE_DIM))
            self._loaded[version] = loaded
            metrics.MODEL_LOADED.set(len(self._loaded))
            return loaded

    # ---- role accessors ------------------------------------------------

    @property
    def primary(self) -> LoadedModel | None:
        return self._primary

    @property
    def shadow(self) -> LoadedModel | None:
        return self._shadow

    @property
    def previous(self) -> LoadedModel | None:
        return self._previous

    def require_primary(self) -> LoadedModel:
        primary = self._primary
        if primary is None:
            raise NoPrimaryError("no primary model loaded")
        return primary

    @property
    def ready(self) -> bool:
        return self._primary is not None

    # ---- swaps -----------------------------------------------------------

    def promote(self, version: str) -> dict:
        """Make `version` the primary. Loads and warms first, then swaps atomically."""
        candidate = self.load(version)
        with self._swap_lock:
            old = self._primary
            if old is not None and old.version == candidate.version:
                return self._swap_record("promote", old, candidate, changed=False)
            self._previous = old
            self._primary = candidate  # single reference assignment: the swap itself
            if self._shadow is not None and self._shadow.version == candidate.version:
                self._shadow = None
            self._refresh_role_gauges()
            metrics.VERSION_SWAPS.labels(kind="promote").inc()
            return self._swap_record("promote", old, candidate, changed=True)

    def rollback(self) -> dict:
        with self._swap_lock:
            if self._previous is None:
                raise NoPrimaryError("nothing to roll back to")
            old, self._primary, self._previous = self._primary, self._previous, self._primary
            self._refresh_role_gauges()
            metrics.VERSION_SWAPS.labels(kind="rollback").inc()
            return self._swap_record("rollback", old, self._primary, changed=True)

    def set_shadow(self, version: str | None) -> LoadedModel | None:
        if version is None:
            with self._swap_lock:
                self._shadow = None
                self._refresh_role_gauges()
            return None
        candidate = self.load(version)
        with self._swap_lock:
            self._shadow = candidate
            self._refresh_role_gauges()
        return candidate

    def _swap_record(self, kind: str, old: LoadedModel | None, new: LoadedModel, changed: bool):
        record = {
            "kind": kind,
            "from": old.version if old else None,
            "to": new.version,
            "changed": changed,
            "at": time.time(),
        }
        if changed:
            self.swap_history.append(record)
        return record

    def _refresh_role_gauges(self) -> None:
        for version in self.available_versions():
            metrics.MODEL_VERSION_INFO.labels(version=version, role="primary").set(
                1 if self._primary and self._primary.version == version else 0
            )
            metrics.MODEL_VERSION_INFO.labels(version=version, role="shadow").set(
                1 if self._shadow and self._shadow.version == version else 0
            )

    # ---- reporting -------------------------------------------------------

    def describe(self) -> dict:
        versions = []
        for version in self.available_versions():
            entry = self._manifest["versions"][version]
            versions.append(
                {
                    "version": version,
                    "file": entry["file"],
                    "test_mae_minutes": entry.get("metrics", {}).get("test_mae_minutes"),
                    "loaded": version in self._loaded,
                    "role": self._role_of(version),
                }
            )
        return {
            "primary": self._primary.version if self._primary else None,
            "shadow": self._shadow.version if self._shadow else None,
            "previous": self._previous.version if self._previous else None,
            "versions": versions,
            "swaps": list(self.swap_history),
        }

    def _role_of(self, version: str) -> str | None:
        if self._primary and self._primary.version == version:
            return "primary"
        if self._shadow and self._shadow.version == version:
            return "shadow"
        return None
