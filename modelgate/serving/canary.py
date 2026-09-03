"""Weighted canary routing with automatic rollback.

A canary is a candidate version that answers a fixed share of live traffic.
Routing is a deterministic weighted split (a fractional accumulator, so a 5%
weight sends exactly 1 in 20 requests to the candidate), and every served
request records its outcome and inference latency into a rolling window for
its version. After each candidate sample the router compares the candidate's
error rate and p95 inference latency against the primary's over the same
window; if either exceeds the primary by the configured margin the canary is
rolled back on the spot: the weight drops to zero and the candidate is
cleared, so the next request already routes to the primary.

A candidate failure never reaches the client. The request path falls back to
the primary answer for that request and counts the failure against the
candidate, so a broken canary produces rollback, not dropped requests.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field

from modelgate.serving import metrics


@dataclass(frozen=True)
class CanaryThresholds:
    window_seconds: float = 60.0
    min_samples: int = 50
    error_rate_delta: float = 0.02
    latency_ratio: float = 2.0
    latency_floor_ms: float = 1.0


@dataclass(frozen=True)
class Sample:
    at: float
    latency_s: float
    ok: bool


@dataclass
class _Window:
    samples: deque[Sample] = field(default_factory=deque)

    def add(self, sample: Sample) -> None:
        self.samples.append(sample)

    def prune(self, now: float, horizon: float) -> None:
        cutoff = now - horizon
        while self.samples and self.samples[0].at < cutoff:
            self.samples.popleft()

    def stats(self) -> dict:
        n = len(self.samples)
        errors = sum(1 for s in self.samples if not s.ok)
        latencies = sorted(s.latency_s for s in self.samples if s.ok)
        return {
            "samples": n,
            "errors": errors,
            "error_rate": round(errors / n, 4) if n else 0.0,
            "p50_ms": round(_percentile(latencies, 50) * 1000, 3),
            "p95_ms": round(_percentile(latencies, 95) * 1000, 3),
        }


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    k = max(0, min(len(values) - 1, math.ceil(pct / 100.0 * len(values)) - 1))
    return values[k]


class CanaryRouter:
    def __init__(self, thresholds: CanaryThresholds, clock=time.time) -> None:
        self.thresholds = thresholds
        self._clock = clock
        self._lock = threading.Lock()
        self._candidate = None
        self._weight = 0.0
        self._accumulator = 0.0
        self._windows: dict[str, _Window] = {}
        self._status = "none"
        self._started_at: float | None = None
        self._last_rollback: dict | None = None
        self.rollback_history: list[dict] = []
        metrics.CANARY_WEIGHT.set(0.0)

    # ---- configuration ---------------------------------------------------

    @property
    def candidate(self):
        return self._candidate

    @property
    def weight(self) -> float:
        return self._weight

    @property
    def status(self) -> str:
        return self._status

    def start(self, candidate, weight: float) -> None:
        if not 0.0 < weight <= 1.0:
            raise ValueError("canary weight must be in (0, 1]")
        with self._lock:
            self._candidate = candidate
            self._weight = weight
            self._accumulator = 0.0
            self._windows = {}
            self._status = "active"
            self._started_at = self._clock()
            self._last_rollback = None
            metrics.CANARY_WEIGHT.set(weight)
            metrics.CANARY_INFO.labels(version=candidate.version).set(1)

    def clear(self) -> None:
        with self._lock:
            self._clear_locked("none")

    def _clear_locked(self, status: str) -> None:
        if self._candidate is not None:
            metrics.CANARY_INFO.labels(version=self._candidate.version).set(0)
        self._candidate = None
        self._weight = 0.0
        self._accumulator = 0.0
        self._status = status
        metrics.CANARY_WEIGHT.set(0.0)

    # ---- routing ---------------------------------------------------------

    def pick(self, primary):
        """Return the candidate for this request, or None to serve the primary."""
        candidate = self._candidate
        if candidate is None or candidate.version == primary.version:
            return None
        with self._lock:
            if self._candidate is not candidate:
                return None
            self._accumulator += self._weight
            if self._accumulator >= 1.0:
                self._accumulator -= 1.0
                return candidate
        return None

    # ---- observation -----------------------------------------------------

    def record(self, version: str, latency_s: float, ok: bool, primary_version: str) -> None:
        """Record one served request. Evaluates rollback after candidate samples."""
        now = self._clock()
        with self._lock:
            window = self._windows.setdefault(version, _Window())
            window.add(Sample(now, latency_s, ok))
            candidate = self._candidate
            if candidate is None or version != candidate.version:
                return
            verdict = self._evaluate_locked(now, candidate.version, primary_version)
            if verdict is not None:
                self._rollback_locked(now, verdict)

    def _evaluate_locked(self, now: float, candidate_version: str, primary_version: str):
        t = self.thresholds
        cand = self._windows.get(candidate_version)
        prim = self._windows.get(primary_version)
        if cand is None or prim is None:
            return None
        cand.prune(now, t.window_seconds)
        prim.prune(now, t.window_seconds)
        cs, ps = cand.stats(), prim.stats()
        if cs["samples"] < t.min_samples or ps["samples"] < t.min_samples:
            return None
        if cs["error_rate"] > ps["error_rate"] + t.error_rate_delta:
            return {
                "reason": "error_rate",
                "candidate": cs["error_rate"],
                "primary": ps["error_rate"],
                "threshold": round(ps["error_rate"] + t.error_rate_delta, 4),
            }
        if cs["p95_ms"] > ps["p95_ms"] * t.latency_ratio and (
            cs["p95_ms"] - ps["p95_ms"] > t.latency_floor_ms
        ):
            return {
                "reason": "latency",
                "candidate": cs["p95_ms"],
                "primary": ps["p95_ms"],
                "threshold": round(max(ps["p95_ms"] * t.latency_ratio, t.latency_floor_ms), 3),
            }
        return None

    def _rollback_locked(self, now: float, verdict: dict) -> None:
        record = {
            "version": self._candidate.version,
            "weight": self._weight,
            "at": now,
            **verdict,
        }
        self._last_rollback = record
        self.rollback_history.append(record)
        metrics.CANARY_ROLLBACKS.labels(reason=verdict["reason"]).inc()
        self._clear_locked("rolled_back")

    # ---- reporting -------------------------------------------------------

    def report(self, primary_version: str | None) -> dict:
        now = self._clock()
        with self._lock:
            candidate = self._candidate
            rollback = self._last_rollback
            versions = {}
            for version, window in self._windows.items():
                window.prune(now, self.thresholds.window_seconds)
                versions[version] = window.stats()
            return {
                "status": self._status,
                "candidate": candidate.version if candidate else None,
                "primary": primary_version,
                "weight": self._weight,
                "started_at": self._started_at,
                "thresholds": {
                    "window_seconds": self.thresholds.window_seconds,
                    "min_samples": self.thresholds.min_samples,
                    "error_rate_delta": self.thresholds.error_rate_delta,
                    "latency_ratio": self.thresholds.latency_ratio,
                    "latency_floor_ms": self.thresholds.latency_floor_ms,
                },
                "versions": versions,
                "last_rollback": rollback,
                "rollbacks": len(self.rollback_history),
                "reported_at": now,
            }
