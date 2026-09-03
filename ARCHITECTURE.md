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

**Warm pool.** `promote()` on a version that is not resident loads it first, which is the
only part of a swap that takes measurable time. `warm()` does that load ahead of time into
a spare slot, so the later promote finds the version resident, records `prewarmed: true`
and `load_seconds: 0`, and goes straight to the reference assignment. The pool keeps at
most `pool_size` versions; when it is over capacity it evicts the least recently touched
version that holds no role. The primary, the previous version (rollback target), the
shadow, and whatever the app pins (the canary candidate) are never evicted, and the version
being warmed is protected during its own warm call. Eviction only removes the cache entry;
a request that captured a reference to an evicted model finishes on it.

`modelgate_dropped_requests_total` counts any `/predict` that ended in a 5xx: an unhandled
exception, or a call with no primary loaded. The load generator counts every request it
sent and reports any non-2xx or connection failure as dropped. Both numbers stay at 0
across the swap in `make demo`; the tests in `tests/test_swap_zero_drop.py` assert the same
in-process with 2000 requests over 50 workers, using both asyncio and threads.

## Micro-batching

`Batcher` in `modelgate/serving/batching.py` queues feature rows per loaded model on the
event loop and runs them in one forward pass. A batch starts when it holds
`max_batch_size` rows or `max_wait_s` after its first row arrived, and queues drain in
FIFO order, so a request waits at most `max_wait_s` plus one batch execution for every
`max_batch_size` rows queued ahead of it. Flushes are scheduled on the loop rather than run
inline in the submitting call, so every waiter is parked on its future before results land
and wakes in submission order; `tests/test_batching.py` asserts the completion order of
100 rows over batches of 8.

Batching only helps when arrivals overlap. A row that arrives more than `max_wait_s` after
the previous one for its model is treated as sparse and runs on the next loop step; rows
within `max_wait_s` of each other are held so they can share a pass. At 200 rps with 5 ms
between arrivals nothing waits, which keeps the single-row p50 at 1.7 ms; a burst of 16
concurrent requests forms batches. For this MLP the forward pass is about 0.2 ms of a
1.5 ms request, so batching does not move throughput much; the mechanism is here for the
day the model is heavier.

**Exactness.** Small-matrix kernels choose different blocking for different row counts, so
`net(x[i:i+1])` and `net(x)[i]` can differ in the last bit; a 2 decimal rounding then
flips for the occasional row. `LoadedModel.predict_batch` therefore pads every forward
pass to exactly `pad_rows` rows (the batch size) with zero rows and slices the result. A
row's output depends only on its own values and the row count, which is now constant, so a
request gets the same bits alone, in a batch of 5, or in a batch of 32. The single-row
`predict()` goes through the same padded path. The tests compare 600 rows over 60 random
batch compositions per version and find no mismatch, and the swap test under batching
checks every one of 2000 responses against the single-row answer of the version that
served it.

Model failures inside a batch set the exception on every waiter in that batch and the
queue keeps draining. For the primary that is an unhandled error and counts as a drop, as
before; for a canary candidate `_serve_canary` catches it and answers from the primary.

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

## Canary routing

A canary is a candidate that answers a share of live traffic, unlike a shadow which only
observes. `CanaryRouter` in `modelgate/serving/canary.py` owns three things.

**Deterministic split.** Routing uses a fractional accumulator under a lock: each request
adds the weight, and when the accumulator reaches 1 it is decremented and the request goes
to the candidate. A 5% weight therefore sends exactly every 20th request to the candidate
instead of a random 5% on average, which makes the split testable to the request and keeps
the candidate's sample count predictable at low traffic. The candidate reference is read
once per request, the same way the primary is, so a rollback mid-request does not change
which model answers.

**Rolling per-version windows.** Every served request records `(at, latency, ok)` for the
version that answered, where latency is the inference call alone rather than the full
request, so both versions are measured on the same request path. After each candidate
sample the router prunes both windows to `window_seconds` and, once both hold at least
`min_samples`, compares error rate and p95. The error rule is additive (`candidate >
primary + delta`); the latency rule is a ratio with an absolute floor, because the ETA MLP
evaluates in a few hundred microseconds and a pure ratio would fire on scheduler noise.

**Rollback in the same call.** When a rule fires, the router records the verdict, bumps
`modelgate_canary_rollbacks_total{reason}`, sets the weight to zero, and clears the
candidate, all under the same lock that `pick()` takes. The very next request routes to the
primary. No background task is involved and there is no window in which a rolled-back
canary can still be chosen.

A candidate failure is caught in `_serve_canary`: the request is answered by the primary,
the failure is counted as `modelgate_canary_requests_total{outcome="fallback"}` and as an
error sample for the candidate, and the client sees a 200 with `model_version` set to the
primary. This is what lets a broken canary trigger rollback without a single drop, which
`tests/test_canary.py` asserts with an always-raising candidate.

Promoting the canary version clears the canary, as with the shadow; a version cannot route
to itself.

## Feature drift

Drift is scored per input feature against statistics the trainer writes into the manifest
(`modelgate/model/stats.py`): for numeric inputs the mean, std, five quantiles, and the
nine decile edges of the training column; for categorical inputs (hour, day of week, zone,
rain) the frequency of each category. Because training is seeded, the reference is
reproducible and `tests/test_drift.py` asserts that recomputing it from the dataset gives
the manifest byte for byte.

`DriftMonitor` in `modelgate/serving/drift.py` keeps one bounded deque per feature of the
values in accepted `/predict` payloads. The score is the population stability index (PSI):
live values are binned into the ten equal-mass bins defined by the training deciles (so the
expected share is 0.1 per bin) or into the training categories, and
`sum((obs - exp) * ln(obs / exp))` is taken with a small floor on both sides so an empty
bin is finite. PSI is 0 for an identical distribution, about 0.1 for a mild shift, and above
0.25 for one that should be looked at; those two thresholds map to the `moderate` and
`drifted` statuses and are configurable. Every feature also reports live mean, std, and
quantiles or frequencies next to the training values, so the direction of a shift is
visible without a second query.

Unknown categories are rejected by validation before a payload reaches the window, which
is the right place for them (the model has no embedding for a new zone) but would make
them invisible to drift. The validation handler therefore forwards `unknown_zone`
rejections to `record_unknown`, and the report shows an unknown-category rate over the
same window as the accepted values.

Scores are recomputed into the `modelgate_feature_drift` gauges every `refresh_every`
observations rather than on every request: the window is sorted for quantiles and PSI, and
doing that at 200 rps for six features would be wasted work. `GET /admin/drift` always
recomputes. No inference uses the drift path, so a bug there cannot change an answer.

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

## Request log and replay

The request log exists so a candidate can be judged on real inputs offline, without
running it as a shadow or canary first. `RequestLog` in `modelgate/serving/reqlog.py`
writes one JSON line per sampled accepted request with exactly four keys: `at`, `input`
(the six validated numeric fields), `version`, and `eta_minutes`. Nothing from the HTTP
layer is written, including the client-supplied request id, so the file has no route for
PII to enter it and can be shared. Sampling uses the same fractional accumulator as the
canary router: a rate of 0.5 writes every second accepted request, which the tests check
by position. Writes take a lock and go to a buffered file flushed every hundred records,
on `GET /admin/request-log`, and at shutdown.

`modelgate eval` (`modelgate/eval.py`) reads the log, validates each record with
`PredictRequest` so that a malformed or out-of-range input is skipped and counted rather
than silently encoded, builds the feature matrix with the shared `encode`, and runs each
version's `predict_batch`. Because that path pads every forward pass to a fixed row count,
the replay produces the same bits the service produced at serving time. The report
exploits that: for every record whose `version` is one of the replayed versions it compares
the replayed answer, rounded as the API rounds, with the logged one. A non-zero mismatch
count means the artifact, the feature code, or the padding changed since the log was
written, which is exactly the kind of drift a replay should surface before a promote.

Accuracy uses whatever truth the log has. Production logs will not carry actuals at
serving time, so `actual_eta_minutes` is an optional field to be joined in later; the
committed fixture has it from the seeded dataset, and `--truth reference` evaluates
against the synthetic formula for a noise-free view. Calibration buckets predictions by
predicted ETA and reports mean predicted against mean actual per bucket, with the expected
calibration error as the sample-weighted absolute gap; that is where the v1 fixture shows
its pattern of over-predicting short trips and under-predicting long ones while v2 sits
within a minute across the range. Divergence between the two versions reuses the shadow
report's statistics so the offline number can be compared directly with the online one.

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
