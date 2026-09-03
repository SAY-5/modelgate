"""Prometheus metrics for the serving layer.

Metrics are created once at import time and shared by every app instance, so
a process exposes one consistent series set no matter how many test apps are
built. Helper accessors exist so tests can assert on values without parsing
the exposition text.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, disable_created_metrics

# The *_created companion series double the exposition size without adding signal here.
disable_created_metrics()

LATENCY_BUCKETS = (
    0.0005,
    0.001,
    0.0025,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
)
ETA_BUCKETS = (1, 2, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 300)
DIVERGENCE_BUCKETS = (0.1, 0.25, 0.5, 1, 2, 3, 5, 10, 20, 60)

REQUESTS = Counter(
    "modelgate_requests_total",
    "Prediction requests by serving model version and outcome.",
    ["version", "outcome"],
)
REQUEST_LATENCY = Histogram(
    "modelgate_request_latency_seconds",
    "End-to-end /predict latency including validation and shadow run.",
    ["version"],
    buckets=LATENCY_BUCKETS,
)
PREDICTIONS_ETA = Histogram(
    "modelgate_predictions_eta_minutes",
    "Distribution of ETA values returned to clients.",
    ["version"],
    buckets=ETA_BUCKETS,
)
INPUT_REJECTIONS = Counter(
    "modelgate_input_rejections_total",
    "Requests rejected by input validation, by reason.",
    ["reason"],
)
SHADOW_DIVERGENCE = Histogram(
    "modelgate_shadow_divergence_minutes",
    "Absolute difference between shadow and primary ETA in minutes.",
    ["primary", "shadow"],
    buckets=DIVERGENCE_BUCKETS,
)
SHADOW_REQUESTS = Counter(
    "modelgate_shadow_requests_total",
    "Shadow inferences executed, by outcome.",
    ["shadow", "outcome"],
)
MODEL_VERSION_INFO = Gauge(
    "modelgate_model_version_info",
    "1 for the version currently serving in the given role (primary or shadow).",
    ["version", "role"],
)
VERSION_SWAPS = Counter(
    "modelgate_version_swaps_total",
    "Primary version swaps (promotions and rollbacks).",
    ["kind"],
)
DROPPED_REQUESTS = Counter(
    "modelgate_dropped_requests_total",
    "Prediction requests that failed for a reason other than invalid input. Must stay 0.",
)
MODEL_LOADED = Gauge(
    "modelgate_models_loaded",
    "Number of model versions resident in memory.",
)
CANARY_WEIGHT = Gauge(
    "modelgate_canary_weight",
    "Share of /predict traffic routed to the canary candidate (0 when no canary is active).",
)
CANARY_INFO = Gauge(
    "modelgate_canary_info",
    "1 for the version currently serving as the canary candidate.",
    ["version"],
)
CANARY_REQUESTS = Counter(
    "modelgate_canary_requests_total",
    "Requests routed to the canary candidate, by outcome (ok, or fallback to the primary).",
    ["version", "outcome"],
)
CANARY_ROLLBACKS = Counter(
    "modelgate_canary_rollbacks_total",
    "Automatic canary rollbacks, by the threshold that fired.",
    ["reason"],
)


def counter_value(counter: Counter, **labels: str) -> float:
    """Read the current value of a counter child (test helper)."""
    child = counter.labels(**labels) if labels else counter
    return child._value.get()


def histogram_count(hist: Histogram, **labels: str) -> float:
    """Number of observations recorded by a histogram child (test helper)."""
    child = hist.labels(**labels) if labels else hist
    # Buckets are stored non-cumulatively; their sum is the observation count.
    return sum(b.get() for b in child._buckets)


def gauge_value(gauge: Gauge, **labels: str) -> float:
    child = gauge.labels(**labels) if labels else gauge
    return child._value.get()
