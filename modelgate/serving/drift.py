"""Per-feature drift monitoring against the training manifest.

Every accepted /predict payload contributes one value per input feature to a
rolling window. Each feature is scored against the reference statistics the
trainer wrote to `manifest.json`:

- numeric features are binned into the ten equal-mass bins defined by the
  training decile edges, and the population stability index (PSI) of the live
  histogram against the uniform training histogram is the drift score;
- categorical features use PSI over category frequencies, plus an
  unknown-category rate fed by validation rejections (values the model was
  never trained on never reach the window, so they are counted separately).

PSI is 0 for an identical distribution, around 0.1 for a mild shift, and above
0.25 for a shift that should be looked at. The window also reports live mean,
std, and quantiles next to the training values so the direction of a shift is
visible at a glance. Scores are refreshed into the `modelgate_feature_drift`
gauges every `refresh_every` observations and on every report.
"""

from __future__ import annotations

import bisect
import math
import threading
import time
from collections import Counter, deque

from modelgate.model.stats import QUANTILES, category_key, mean_std, quantile
from modelgate.serving import metrics

_EPS = 1e-4


def psi(observed: list[float], expected: list[float]) -> float:
    """Population stability index between two aligned probability vectors."""
    total = 0.0
    for obs, exp in zip(observed, expected, strict=True):
        o = max(obs, _EPS)
        e = max(exp, _EPS)
        total += (o - e) * math.log(o / e)
    return total


class _FeatureWindow:
    def __init__(self, name: str, reference: dict, window_size: int) -> None:
        self.name = name
        self.reference = reference
        self.kind = reference["kind"]
        self.values: deque = deque(maxlen=window_size)
        self.unknown: deque[int] = deque(maxlen=window_size)

    def observe(self, value) -> None:
        self.values.append(value)
        self.unknown.append(0)

    def record_unknown(self) -> None:
        self.unknown.append(1)

    def score(self) -> tuple[float, dict]:
        values = list(self.values)
        if self.kind == "numeric":
            return self._score_numeric(values)
        return self._score_categorical(values)

    def _score_numeric(self, values: list) -> tuple[float, dict]:
        ordered = sorted(float(v) for v in values)
        edges = self.reference["bin_edges"]
        counts = [0] * (len(edges) + 1)
        for v in ordered:
            counts[bisect.bisect_right(edges, v)] += 1
        n = len(ordered)
        observed = [c / n for c in counts] if n else [0.0] * len(counts)
        expected = [1.0 / len(counts)] * len(counts)
        mean, std = mean_std(ordered)
        return psi(observed, expected), {
            "mean": round(mean, 4),
            "std": round(std, 4),
            "quantiles": {f"p{p}": round(quantile(ordered, p), 4) for p in QUANTILES},
            "bins": [round(o, 4) for o in observed],
        }

    def _score_categorical(self, values: list) -> tuple[float, dict]:
        freq = self.reference["frequencies"]
        counts = Counter(category_key(v) for v in values)
        n = len(values)
        keys = list(freq)
        observed = [counts.get(k, 0) / n if n else 0.0 for k in keys]
        expected = [freq[k] for k in keys]
        numeric = [float(v) for v in values]
        mean, std = mean_std(numeric)
        return psi(observed, expected), {
            "mean": round(mean, 4),
            "std": round(std, 4),
            "frequencies": {k: round(o, 4) for k, o in zip(keys, observed, strict=True)},
        }


class DriftMonitor:
    def __init__(
        self,
        training_stats: dict | None,
        window_size: int = 2000,
        min_samples: int = 100,
        warn_threshold: float = 0.1,
        alert_threshold: float = 0.25,
        refresh_every: int = 100,
    ) -> None:
        self.enabled = bool(training_stats)
        self.window_size = window_size
        self.min_samples = min_samples
        self.warn_threshold = warn_threshold
        self.alert_threshold = alert_threshold
        self.refresh_every = max(1, refresh_every)
        self._lock = threading.Lock()
        self._features: dict[str, _FeatureWindow] = {}
        self._observed = 0
        self._last_refresh_at: float | None = None
        self._last_scores: dict[str, float] = {}
        if self.enabled:
            for name, ref in training_stats["features"].items():
                self._features[name] = _FeatureWindow(name, ref, window_size)

    def observe(self, payload: dict) -> None:
        if not self.enabled:
            return
        with self._lock:
            for name, window in self._features.items():
                if name in payload:
                    window.observe(payload[name])
            self._observed += 1
            if self._observed % self.refresh_every == 0:
                self._refresh_locked()

    def record_unknown(self, feature: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            window = self._features.get(feature)
            if window is not None:
                window.record_unknown()

    def reset(self) -> None:
        with self._lock:
            for window in self._features.values():
                window.values.clear()
                window.unknown.clear()
            self._observed = 0
            self._last_scores = {}
            for name in self._features:
                metrics.FEATURE_DRIFT.labels(feature=name).set(0.0)
                metrics.FEATURE_UNKNOWN_RATE.labels(feature=name).set(0.0)
            metrics.DRIFT_SAMPLES.set(0)

    def _status(self, score: float, samples: int) -> str:
        if samples < self.min_samples:
            return "insufficient"
        if score >= self.alert_threshold:
            return "drifted"
        if score >= self.warn_threshold:
            return "moderate"
        return "stable"

    def _refresh_locked(self) -> dict:
        features: dict[str, dict] = {}
        for name, window in self._features.items():
            samples = len(window.values)
            score, live = window.score()
            unknown = sum(window.unknown) / len(window.unknown) if window.unknown else 0.0
            status = self._status(score, samples)
            reported = round(score, 4) if samples >= self.min_samples else 0.0
            metrics.FEATURE_DRIFT.labels(feature=name).set(reported)
            metrics.FEATURE_UNKNOWN_RATE.labels(feature=name).set(round(unknown, 4))
            self._last_scores[name] = reported
            ref = window.reference
            reference = {"mean": ref["mean"], "std": ref["std"]}
            if window.kind == "numeric":
                reference["quantiles"] = ref["quantiles"]
            else:
                reference["frequencies"] = ref["frequencies"]
            features[name] = {
                "kind": window.kind,
                "samples": samples,
                "status": status,
                "drift_score": round(score, 4),
                "unknown_rate": round(unknown, 4),
                "live": live,
                "reference": reference,
            }
        metrics.DRIFT_SAMPLES.set(min(self._observed, self.window_size))
        self._last_refresh_at = time.time()
        return features

    def report(self) -> dict:
        if not self.enabled:
            return {"enabled": False, "features": {}}
        with self._lock:
            features = self._refresh_locked()
            worst = max(features.values(), key=lambda f: f["drift_score"], default=None)
            return {
                "enabled": True,
                "window_size": self.window_size,
                "min_samples": self.min_samples,
                "thresholds": {"warn": self.warn_threshold, "alert": self.alert_threshold},
                "observed": self._observed,
                "drifted": sorted(n for n, f in features.items() if f["status"] == "drifted"),
                "worst_feature": (
                    next(n for n, f in features.items() if f is worst) if worst else None
                ),
                "features": features,
                "reported_at": self._last_refresh_at,
            }
