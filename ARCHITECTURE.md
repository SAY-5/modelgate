# Architecture

## Zero-drop version swap

The requirement is that promoting a new model version never fails, delays, or corrupts a
request that is already being served. Three pieces make that hold.

**Load and warm off the request path.** `ModelRegistry.load()` reads the artifact, builds
the network, and runs one dummy forward pass so lazy kernel initialisation is paid before
the version can be selected. The admin handler calls it through `asyncio.to_thread`, so the
event loop keeps serving while the file is read. If loading fails, the exception propagates
to the admin caller and the primary is untouched.

**Swap is one reference assignment.** `promote()` takes `_swap_lock`, stores the old
primary in `_previous`, assigns `_primary = candidate`, refreshes the role gauges, and
releases the lock. Nothing inside the critical section does I/O or allocates. The lock
serialises concurrent promote/rollback calls; it is not needed for readers because a Python
attribute store is atomic under the interpreter lock.

**Requests capture their model once.** `/predict` reads `registry.require_primary()` into a
local at the top and uses that `LoadedModel` for the whole request: inference, metrics
labels, and the response body. A swap that happens after that read does not affect the
request. A `LoadedModel` is a frozen dataclass and the network is in eval mode, so sharing
one instance across concurrent requests is safe; PyTorch inference does not mutate module
state.

Rollback is the same swap with the roles of `_primary` and `_previous` exchanged, so it is
instant and does not reload anything.

`modelgate_dropped_requests_total` counts any `/predict` that ended in a 5xx: an unhandled
exception, or a call with no primary loaded. The load generator counts every request it
sent and reports any non-2xx or connection failure as dropped. Both numbers stay at 0
across the swap in `make demo`; the tests in `tests/test_swap_zero_drop.py` assert the same
in-process with 2000 requests over 50 workers, using both asyncio and threads.

## Shadow runs

A shadow version is a candidate you want to observe on production traffic without letting
it answer. When one is set, `/predict` runs the primary first, then calls the shadow with the
same feature tensor inside a `try` block. The shadow result is used only to:

- observe `modelgate_shadow_divergence_minutes{primary, shadow}` with the absolute delta,
- increment `modelgate_shadow_requests_total{shadow, outcome}`,
- append a `ShadowRecord` to a bounded rolling window in `ShadowTracker`.

The response is built from the primary's value before the shadow runs, and any shadow
exception is logged and counted, never raised. `GET /admin/shadow/report` summarises the
window: count, mean/p50/p95/max absolute delta, mean and p95 relative delta, share of
records beyond a configurable threshold, and signed bias (shadow minus primary). The window
resets when the shadow version changes so the report always describes one comparison.

The shadow runs inline rather than in a background task. The model is a 19-input MLP that
evaluates in well under a millisecond, so the extra latency is smaller than the cost of
scheduling a task, and inline execution keeps the divergence record tied to the exact
request. If a heavier shadow model is introduced, the `_run_shadow` helper is the single
place to move to a queue.

Promoting the shadow version clears the shadow slot, since a version cannot be both.

## Input validation

`PredictRequest` is a pydantic model with `strict=True` and `extra="forbid"`:

- strict mode rejects type coercion, so `"5"` is not an int and `1` is not a bool;
- `allow_inf_nan=False` on both float fields rejects `NaN`, `Infinity`, `-Infinity`,
  which JSON parsers otherwise accept and which would silently propagate through the MLP;
- range constraints match the manifest's feature schema (0..500 km, 0..23, 0..6, 0..1);
- the zone id is checked against the zone set the model was trained on, so a new zone
  cannot reach the embedding without a retrain;
- unknown fields are rejected rather than ignored, which catches renamed or misspelled
  client fields early.

The exception handler maps pydantic error types to a small fixed set of reasons
(`out_of_range`, `not_finite`, `wrong_type`, `unknown_zone`, `unknown_field`,
`missing_field`, `malformed_body`) so `modelgate_input_rejections_total{reason}` stays low
cardinality. Rejections count as `outcome="rejected"` in the request counter, not as drops.

Feature encoding lives in `modelgate/model/features.py` and is imported by both training
and serving. Hour and day of week are encoded as sin/cos pairs so that 23:00 and 00:00 are
neighbours, distance is scaled, and the zone is one-hot. `manifest.json` records the
feature names and dimension so a mismatch between artifact and code is visible.

## Metrics design

Metrics are module-level singletons in `modelgate/serving/metrics.py` and use the default
registry. This keeps one series set per process regardless of how many app objects exist,
which matters for tests and for uvicorn's import path.

- Latency and ETA are histograms with buckets chosen for sub-millisecond to seconds latency
  and 1 to 300 minute ETAs, so `histogram_quantile` in Grafana is meaningful without tuning.
- Per-version labels on the request, latency, and ETA series make a swap visible as one
  series ending and another starting in the same scrape.
- `modelgate_model_version_info{version, role}` is a gauge set to 1 for the active version
  in each role and 0 for the others, so a dashboard stat can show the current primary by name
  and a swap shows as a step.
- `_created` companion series are disabled to halve the exposition size.
- The dropped-request counter has no labels. It is the one number that should always read 0.

The Grafana dashboard in `monitoring/grafana/dashboards/modelgate.json` is provisioned from
file; the Prometheus datasource is provisioned alongside it, so `docker compose up` gives a
working dashboard with no clicks.

## Training

`modelgate/model/train.py` builds a 12,000 trip synthetic dataset from a seed using a
non-linear reference formula (speed drops with traffic and rain, night and weekend trips are
faster, each zone adds a pickup delay) plus Gaussian noise, then trains two versions:

- v1: 2 hidden layers of 64, 10 epochs, Adam with cosine decay;
- v2: 3 hidden layers of 96, 25 epochs.

Both are exported as `state_dict` payloads with the architecture embedded, loaded with
`weights_only=True`. The manifest records held-out MAE for each version and for two
baselines (mean predictor, linear fit on distance). `tests/test_training.py` trains twice in
temporary directories and asserts identical MAE and weights, and that both versions beat
both baselines.
