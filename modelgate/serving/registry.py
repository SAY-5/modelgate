"""Model registry with atomic, zero-drop version swaps and a warm pool.

A version is loaded from disk and warmed with dummy inferences off the
request path. Only after that succeeds is the primary reference replaced, and
the replacement is a single attribute assignment taken under a lock. Every
request captures its own reference to the primary at the start, so a request
in flight during a swap finishes on the model it started with. No request
observes a partially loaded model.

The pool keeps up to `pool_size` versions resident. `warm()` loads a version
into a spare slot ahead of time so a later `promote()` finds it already warm
and swaps without touching disk. Versions holding a role (primary, previous,
shadow, or anything the app pins, such as a canary) are never evicted; when
the pool is over capacity the least recently touched unpinned version is
dropped. Requests that still hold a reference to an evicted model finish on
it; eviction only removes the cache entry.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch

from modelgate.model.features import FEATURE_DIM
from modelgate.model.net import EtaNet, load_model
from modelgate.serving import metrics

DEFAULT_PAD_ROWS = 32


class UnknownVersionError(KeyError):
    pass


class NoPrimaryError(RuntimeError):
    pass


@dataclass(frozen=True)
class LoadedModel:
    version: str
    net: EtaNet
    test_mae: float | None
    pad_rows: int = DEFAULT_PAD_ROWS
    loaded_at: float = field(default_factory=time.time)

    def predict_batch(self, features: torch.Tensor) -> list[float]:
        """Run every forward pass at exactly `pad_rows` rows.

        Small-matrix kernels pick different blocking for different row counts,
        which changes results in the last bit. Padding to a fixed size means a
        row gets the same answer alone or inside a full batch.
        """
        out: list[float] = []
        with torch.inference_mode():
            for start in range(0, len(features), self.pad_rows):
                chunk = features[start : start + self.pad_rows]
                buf = torch.zeros(self.pad_rows, features.shape[1], dtype=features.dtype)
                buf[: len(chunk)] = chunk
                out.extend(float(v) for v in self.net(buf)[: len(chunk)])
        return out

    def predict(self, features: torch.Tensor) -> float:
        return self.predict_batch(features)[0]


class ModelRegistry:
    def __init__(
        self,
        artifacts_dir: Path,
        pool_size: int = 3,
        pad_rows: int = DEFAULT_PAD_ROWS,
        extra_pins: Callable[[], set[str]] | None = None,
    ) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.pool_size = max(1, pool_size)
        self.pad_rows = max(1, pad_rows)
        self.extra_pins = extra_pins or (lambda: set())
        self._manifest = self._read_manifest()
        self._loaded: dict[str, LoadedModel] = {}
        self._touched: dict[str, float] = {}
        self._primary: LoadedModel | None = None
        self._previous: LoadedModel | None = None
        self._shadow: LoadedModel | None = None
        self._swap_lock = threading.Lock()
        self._load_lock = threading.Lock()
        self.swap_history: list[dict] = []
        metrics.MODEL_POOL_SLOTS.set(self.pool_size)

    # ---- discovery -------------------------------------------------------

    def _read_manifest(self) -> dict:
        path = self.artifacts_dir / "manifest.json"
        if not path.exists():
            return {"versions": {}}
        return json.loads(path.read_text())

    @property
    def manifest(self) -> dict:
        return self._manifest

    def available_versions(self) -> list[str]:
        return sorted(self._manifest.get("versions", {}))

    def is_known(self, version: str) -> bool:
        return version in self._manifest.get("versions", {})

    def resident_versions(self) -> list[str]:
        return sorted(self._loaded)

    def is_resident(self, version: str) -> bool:
        return version in self._loaded

    # ---- loading ---------------------------------------------------------

    def load(self, version: str) -> LoadedModel:
        """Load and warm a version. Idempotent; safe to call from any thread."""
        if not self.is_known(version):
            raise UnknownVersionError(version)
        cached = self._loaded.get(version)
        if cached is not None:
            self._touched[version] = time.time()
            return cached
        with self._load_lock:
            cached = self._loaded.get(version)
            if cached is not None:
                self._touched[version] = time.time()
                return cached
            entry = self._manifest["versions"][version]
            net = load_model(self.artifacts_dir / entry["file"])
            net.eval()
            loaded = LoadedModel(
                version=version,
                net=net,
                test_mae=entry.get("metrics", {}).get("test_mae_minutes"),
                pad_rows=self.pad_rows,
            )
            # Warm-up: the first forward pass pays for lazy kernel init. Run both a
            # single row and a full padded batch so neither path is cold.
            loaded.predict(torch.zeros(1, FEATURE_DIM))
            loaded.predict_batch(torch.zeros(self.pad_rows, FEATURE_DIM))
            self._loaded[version] = loaded
            self._touched[version] = time.time()
            metrics.MODEL_LOADED.set(len(self._loaded))
            return loaded

    def warm(self, version: str) -> dict:
        """Load `version` into a spare pool slot ahead of a promote."""
        started = time.perf_counter()
        already = self.is_resident(version)
        self.load(version)
        load_seconds = 0.0 if already else time.perf_counter() - started
        with self._swap_lock:
            evicted = self._evict_locked(protect={version})
        metrics.WARM_LOADS.labels(hit="true" if already else "false").inc()
        return {
            "version": version,
            "loaded": not already,
            "load_seconds": round(load_seconds, 4),
            "evicted": evicted,
            "resident": self.resident_versions(),
        }

    def _pinned(self) -> set[str]:
        pins = set(self.extra_pins())
        for model in (self._primary, self._previous, self._shadow):
            if model is not None:
                pins.add(model.version)
        return pins

    def _evict_locked(self, protect: set[str] | None = None) -> list[str]:
        """Drop least recently touched unpinned versions until the pool fits."""
        evicted: list[str] = []
        while len(self._loaded) > self.pool_size:
            pinned = self._pinned() | (protect or set())
            candidates = [v for v in self._loaded if v not in pinned]
            if not candidates:
                break
            victim = min(candidates, key=lambda v: self._touched.get(v, 0.0))
            del self._loaded[victim]
            self._touched.pop(victim, None)
            evicted.append(victim)
            metrics.MODEL_EVICTIONS.inc()
        metrics.MODEL_LOADED.set(len(self._loaded))
        return evicted

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
        started = time.perf_counter()
        prewarmed = self.is_resident(version)
        candidate = self.load(version)
        load_seconds = 0.0 if prewarmed else time.perf_counter() - started
        metrics.SWAP_LOAD_SECONDS.labels(prewarmed="true" if prewarmed else "false").observe(
            load_seconds
        )
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
            record = self._swap_record(
                "promote",
                old,
                candidate,
                changed=True,
                prewarmed=prewarmed,
                load_seconds=round(load_seconds, 4),
            )
            self._evict_locked()
            return record

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
            self._evict_locked()
        return candidate

    def _swap_record(
        self, kind: str, old: LoadedModel | None, new: LoadedModel, changed: bool, **extra
    ):
        record = {
            "kind": kind,
            "from": old.version if old else None,
            "to": new.version,
            "changed": changed,
            "at": time.time(),
            **extra,
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
            "pool": {
                "size": self.pool_size,
                "resident": self.resident_versions(),
                "pinned": sorted(self._pinned()),
            },
            "versions": versions,
            "swaps": list(self.swap_history),
        }

    def _role_of(self, version: str) -> str | None:
        if self._primary and self._primary.version == version:
            return "primary"
        if self._shadow and self._shadow.version == version:
            return "shadow"
        return None
