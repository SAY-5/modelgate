"""Sampled request log for offline replay.

When enabled, a deterministic share of accepted /predict calls is appended to
a JSON-lines file as `{"at", "input", "version", "eta_minutes"}`. `input` holds
the six validated numeric fields the model consumed and nothing else: no
request id, headers, client address, or free text ever reach the file, so it
can be handed to `modelgate eval` without a PII review.

Sampling uses a fractional accumulator like the canary router, so a rate of
0.1 writes exactly every tenth accepted request rather than a random tenth.
Writes go through a lock to a buffered file and are flushed every
`flush_every` records and on close.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from modelgate.serving import metrics

INPUT_FIELDS: tuple[str, ...] = (
    "distance_km",
    "hour_of_day",
    "day_of_week",
    "pickup_zone_id",
    "traffic_index",
    "is_raining",
)


class RequestLog:
    def __init__(self, path: Path | None, sample_rate: float = 1.0, flush_every: int = 100):
        self.path = Path(path) if path else None
        self.enabled = self.path is not None and sample_rate > 0
        self.sample_rate = min(max(sample_rate, 0.0), 1.0)
        self.flush_every = max(1, flush_every)
        self._lock = threading.Lock()
        self._accumulator = 0.0
        self._file = None
        self._since_flush = 0
        self.records = 0
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self.path, "a", encoding="utf-8")  # noqa: SIM115

    def record(self, payload: dict, version: str, eta_minutes: float) -> bool:
        """Append one record if it falls in the sample. Returns whether it was written."""
        if not self.enabled:
            return False
        with self._lock:
            self._accumulator += self.sample_rate
            if self._accumulator < 1.0:
                return False
            self._accumulator -= 1.0
            line = json.dumps(
                {
                    "at": round(time.time(), 3),
                    "input": {k: payload[k] for k in INPUT_FIELDS},
                    "version": version,
                    "eta_minutes": eta_minutes,
                },
                separators=(",", ":"),
            )
            self._file.write(line + "\n")
            self.records += 1
            self._since_flush += 1
            if self._since_flush >= self.flush_every:
                self._file.flush()
                self._since_flush = 0
        metrics.REQUEST_LOG_RECORDS.inc()
        return True

    def flush(self) -> None:
        with self._lock:
            if self._file is not None:
                self._file.flush()
                self._since_flush = 0

    def close(self) -> None:
        with self._lock:
            if self._file is not None:
                self._file.flush()
                self._file.close()
                self._file = None
            self.enabled = False

    def describe(self) -> dict:
        return {
            "enabled": self.enabled,
            "path": str(self.path) if self.path else None,
            "sample_rate": self.sample_rate if self.path else 0.0,
            "records": self.records,
        }
