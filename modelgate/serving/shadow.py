"""Shadow-run bookkeeping: divergence stats between primary and shadow."""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class ShadowRecord:
    request_id: str
    primary_version: str
    shadow_version: str
    primary_eta: float
    shadow_eta: float
    at: float

    @property
    def abs_delta(self) -> float:
        return abs(self.shadow_eta - self.primary_eta)

    @property
    def rel_delta(self) -> float:
        return self.abs_delta / max(self.primary_eta, 1e-6)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, math.ceil(pct / 100.0 * len(ordered)) - 1))
    return ordered[k]


class ShadowTracker:
    def __init__(self, threshold_minutes: float, max_records: int) -> None:
        self.threshold_minutes = threshold_minutes
        self._records: deque[ShadowRecord] = deque(maxlen=max_records)
        self._lock = threading.Lock()
        self._total = 0
        self._errors = 0

    def record(self, rec: ShadowRecord) -> None:
        with self._lock:
            self._records.append(rec)
            self._total += 1

    def record_error(self) -> None:
        with self._lock:
            self._errors += 1

    def reset(self) -> None:
        with self._lock:
            self._records.clear()
            self._total = 0
            self._errors = 0

    def report(self) -> dict:
        with self._lock:
            records = list(self._records)
            total, errors = self._total, self._errors
        abs_deltas = [r.abs_delta for r in records]
        rel_deltas = [r.rel_delta for r in records]
        beyond = sum(1 for d in abs_deltas if d > self.threshold_minutes)
        n = len(records)
        return {
            "count": total,
            "errors": errors,
            "window": n,
            "threshold_minutes": self.threshold_minutes,
            "abs_delta_minutes": {
                "mean": round(sum(abs_deltas) / n, 4) if n else 0.0,
                "p50": round(_percentile(abs_deltas, 50), 4),
                "p95": round(_percentile(abs_deltas, 95), 4),
                "max": round(max(abs_deltas), 4) if n else 0.0,
            },
            "rel_delta": {
                "mean": round(sum(rel_deltas) / n, 4) if n else 0.0,
                "p95": round(_percentile(rel_deltas, 95), 4),
            },
            "share_beyond_threshold": round(beyond / n, 4) if n else 0.0,
            "shadow_bias_minutes": (
                round(sum(r.shadow_eta - r.primary_eta for r in records) / n, 4) if n else 0.0
            ),
            "last": (
                {
                    "request_id": records[-1].request_id,
                    "primary": records[-1].primary_eta,
                    "shadow": records[-1].shadow_eta,
                    "at": records[-1].at,
                }
                if records
                else None
            ),
            "reported_at": time.time(),
        }
