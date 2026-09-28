# ModelGate

ML model serving and monitoring for a PyTorch ETA model. A FastAPI service with strict
input validation, shadow runs and weighted canaries for candidate versions with automatic
rollback, per-feature drift scores against the training manifest, micro-batched inference
with a warm model pool, a sampled request log with an offline replay harness, Prometheus
metrics on a provisioned Grafana dashboard, and load-tested version swaps that complete
with 0 dropped requests.

Python 3.12, PyTorch (CPU), FastAPI, prometheus_client, Grafana, uv.

## What it does

- Serves `POST /predict` for a small MLP that estimates trip ETA in minutes from distance,
  time of day, day of week, pickup zone, traffic index, and rain.
- Validates every input before a tensor is built: types, ranges, known zones, finite floats,
  no unknown fields. Rejections return 422 with a per-field reason and are counted by reason.
- Keeps a model registry with a primary and an optional shadow version. Shadow inference runs
  on every request, divergence against the primary is recorded, and the client always gets
  the primary answer.
- Routes a weighted share of live traffic to a canary version. Per-version error rate and
  p95 inference latency are tracked over a rolling window, and the canary is rolled back
  automatically when it exceeds the primary by a configured margin. A canary failure falls
  back to the primary answer for that request, so a broken candidate never drops a request.
- Scores input drift per feature. The trainer writes mean, std, quantiles, decile edges,
  and category frequencies for every input into `manifest.json`; serving keeps a rolling
  window of accepted inputs and reports a population stability index per feature, the
  unknown-category rate, and live versus training statistics on `GET /admin/drift` and as
  `modelgate_feature_drift` gauges.
- Micro-batches concurrent `/predict` calls. Requests for the same model are queued on the
  event loop and run in one forward pass, up to a maximum batch size or a maximum wait,
  FIFO. Every forward pass is padded to a fixed row count, so a request gets bit-identical
  output alone or in a full batch. Sparse traffic skips the wait entirely.
- Keeps a warm pool of resident versions. `POST /admin/warm` loads and warms a candidate
  into a spare slot so the later promote swaps in 0 s without touching disk; the pool
  evicts the least recently used version that holds no role when it is over capacity.
- Captures a sampled, PII-free request log (the six numeric inputs, the serving version,
  and the answer) behind a flag, and replays it offline: `modelgate eval` runs the log
  against any two versions and reports MAE, a calibration table, and their divergence,
  deterministically and bit-identical to what the service answered.
- Swaps versions atomically. The candidate is loaded and warmed off the request path, then
  the primary reference is replaced in O(1) under a lock. In-flight requests finish on the
  model they started with. `make demo` proves this with a 200 rps load test that promotes
  v2 mid-run.
- Exposes Prometheus metrics for traffic, latency, ETA distribution, rejections, shadow
  divergence, swaps, and dropped requests, with a Grafana dashboard provisioned from
  `monitoring/`.

## Architecture

```
                         +--------------------------------------------------+
  clients                |  FastAPI (modelgate.serving.app)                 |
  POST /predict  ------> |  pydantic strict schema -> 422 + rejection metric |
                         |        |                                         |
                         |        v                                         |
                         |  Batcher: FIFO micro-batches per model, padded   |
                         |        |                                         |
                         |        v                                         |
                         |  DriftMonitor: rolling window vs manifest stats  |
                         |        |                                         |
                         |        v                                         |
                         |  CanaryRouter: weighted split, auto-rollback     |
                         |        |                                         |
                         |        v                                         |
                         |  ModelRegistry                                   |
                         |    primary ---> LoadedModel(v1)  --> response    |
                         |    canary  ---> LoadedModel(v2)  --> response    |
                         |    shadow  ---> LoadedModel(v2)  --> divergence  |
                         |    previous     (rollback target)    log+metrics |
                         |    warm pool    (spare slots, LRU eviction)      |
                         |        ^                                         |
  admin (X-Admin-Token)  |        | load + warm in thread, swap under lock   |
  POST /admin/promote -> |  /admin/shadow /admin/canary /admin/drift ...    |
                         |                                                  |
  Prometheus  <--------- |  GET /metrics      GET /healthz   GET /readyz    |
                         +--------------------------------------------------+
        |
        v
  Grafana (dashboard auto-provisioned from monitoring/grafana)

  artifacts/eta_v1.pt, eta_v2.pt, manifest.json   <--  modelgate.model.train (seeded)
  requests.jsonl (sampled, numeric inputs only)   -->  modelgate eval --versions v1 v2
```

## Quick start

```bash
uv sync --extra dev          # CPU torch from the PyTorch wheel index
make test                    # includes startup, Compose configuration and zero-drop swap tests
make demo                    # start the server, 200 rps for 20 s, promote v2 at t+10 s
```

The configuration tests require Docker Compose (no running daemon is needed). `make serve`
starts the API on `127.0.0.1:8000`. When `MODELGATE_ADMIN_TOKEN` is unset, the launcher uses
`dev-token` for the local demo; explicitly setting it to an empty string disables the admin
API (503). To select a different local port, use `make serve PORT=8001`.

`make demo` output from this machine (Apple M-series, single uvicorn worker):

```
================================================================
ModelGate load test: version swap under load
================================================================
target rate        200.0 rps for 20.0 s
achieved rate      200.0 rps
total requests     4000
successes (2xx)    4000
dropped requests   0  (non-2xx or no response)
server dropped ctr 0.0  (modelgate_dropped_requests_total)
latency ms         p50 1.78  p95 4.18  p99 11.8  max 63.91
shadow enabled     t+5.001s (v2 shadowing)
swap v1 -> v2     requested t+10.0s, completed t+10.006s (2026-09-02 21:43:17)
versions before    {'v1': 2001, 'v2': 1}
versions after     {'v2': 1998}
per-second split   0s:{'v1': 200}  1s:{'v1': 200}  2s:{'v1': 200}  3s:{'v1': 200}  4s:{'v1': 200}  5s:{'v1': 200}  6s:{'v1': 200}  7s:{'v1': 200}  8s:{'v1': 200}  9s:{'v1': 200}  10s:{'v1': 1, 'v2': 199}  11s:{'v2': 200}  12s:{'v2': 200}  13s:{'v2': 200}  14s:{'v2': 200}  15s:{'v2': 200}  16s:{'v2': 200}  17s:{'v2': 200}  18s:{'v2': 200}  19s:{'v2': 200}
shadow report      n=1000 mean|d|=2.2859 p95|d|=6.3505 beyond 2.0min=0.401
================================================================
PASS: 0 dropped requests across the swap
================================================================
```

The one `v2` counted "before" the swap was in flight when the promote completed and was
served by the new primary; the one `v1` in the 10 s bucket was in flight when the swap
happened and finished on the reference it had captured. Nothing errored, nothing hung.

The load generator is open-loop (it sends on schedule regardless of responses), so a slow
or failing server shows up as dropped requests rather than as a lower request rate.

### Replaying a request log

```bash
MODELGATE_REQUEST_LOG=requests.jsonl MODELGATE_REQUEST_LOG_SAMPLE_RATE=0.1 make serve
# ... serve traffic ...
uv run modelgate eval --log requests.jsonl --versions v1 v2 --json report.json
```

Records carry only the validated numeric inputs, the version that answered, and the
answer. The harness validates and encodes them with the same code the API uses and runs
each version's padded forward pass, so the replay reproduces the logged answers bit for
bit (`logged vs replayed: mismatches 0`) and two runs give the same report. Truth comes
from an `actual_eta_minutes` field when a record has one (`--truth auto`), from the
synthetic reference formula (`--truth reference`), or is skipped (`--truth none`). On the
committed 300-record fixture:

```
v1     n=300  MAE 2.9394 min  p95|err| 7.3339  bias 0.0546  within 2 min 0.4367  ECE 2.1494
v2     n=300  MAE 2.0095 min  p95|err| 4.2841  bias 0.4286  within 2 min 0.6433  ECE 0.5655
logged vs replayed: checked 300, mismatches 0, max gap 0.0 min
divergence v1 -> v2: n=300 mean|d| 2.6887 p95|d| 9.0694 max 33.8347 beyond 2.0 min 0.42
```

The calibration table under each version lists mean predicted against mean actual per
predicted-ETA bucket; ECE is the sample-weighted mean absolute gap.

### Running the full stack

```bash
docker compose up --build
# service    http://localhost:8000/docs
# prometheus http://localhost:9090
# grafana    http://localhost:3000  (dashboard "ModelGate: ETA model serving", anonymous viewer)

# drive traffic against the container and swap versions mid-run
uv run python -m loadtest.run --url http://localhost:8000 --rps 200 --duration 60 --shadow-first
```

Compose publishes all three ports on `127.0.0.1` by default. Inside the Docker network,
the API still listens on `0.0.0.0:8000`, so Prometheus can scrape `modelgate:8000`.
Prometheus and Grafana remain loopback-only even when remote API access is enabled.
The bundled Grafana `admin` / `admin` login is for this private local demo only.

#### Deliberate remote API testing

Set `MODELGATE_ADMIN_TOKEN` to a unique, caller-managed random token through your shell's
environment or secret-management tool before using either command:

```bash
# Refuses an unset, empty, whitespace-only, or known development token before listening.
MODELGATE_HOST=0.0.0.0 make serve
# Or publish only the container's API remotely; observability stays on loopback.
MODELGATE_HOST=0.0.0.0 docker compose up --build
```

`MODELGATE_HOST` selects the native listener or the Compose **host publication** address.
Only literal loopback IPs are trusted for development credentials; hostname aliases such as
`localhost` require a configured non-default token too. Validation checks the publication
address, not the container's internal listener. Direct Uvicorn or custom Docker invocations
bypass this launcher and are your responsibility.

This is an explicit development opt-in, not a production deployment configuration. Admin
tokens protect `/admin/*`, not predictions, metrics, or health endpoints. Use TLS, a trusted
network, appropriate firewall rules and rate limits before making the API reachable outside
your host. Neither this token check nor a private bind substitutes for those controls.

For remote observability, keep the ports private and use authenticated SSH forwarding:

```bash
ssh -N -L 127.0.0.1:3000:127.0.0.1:3000 -L 127.0.0.1:9090:127.0.0.1:9090 user@your-host
```

Then open the same local Grafana/Prometheus URLs. Compose configuration checks in this
repository use an isolated environment and `--env-file /dev/null`; they never load your `.env`.

### Training

```bash
make train      # rebuilds artifacts/ from the seeded synthetic dataset in a few seconds
```

Two versions are trained on 12,000 synthetic trips (9,600 train / 2,400 held out):

| version | architecture       | epochs | held-out MAE (min) |
|---------|--------------------|--------|--------------------|
| v1      | MLP 19-64-64-1     | 10     | 3.105              |
| v2      | MLP 19-96-96-96-1  | 25     | 1.888              |

Baselines on the same split: predict the mean, MAE 8.651; linear fit on distance only,
MAE 3.565. Training is deterministic for a given seed; the test suite trains twice and
checks that the weights and MAE match.

The committed artifacts regenerate from their seed, but not bit for bit on a different CPU
architecture: a value sitting on a rounding boundary lands one unit apart between the two hosts.
`tests/fixtures/replay_log.jsonl`, made on arm64, records `distance_km` 8.499 where the x86-64 CI
runner regenerates 8.5, and a six decimal statistic can land one unit apart. The tests that
compare a regenerated artifact against a committed one therefore hold each number to the
tolerance declared for its field: 0.001 for `distance_km` and 0.0001 for `traffic_index`, one
unit in the last place the generator rounds them to; 0.000001 for the training statistics, which
`modelgate/model/stats.py` rounds to six decimals; and 0.02 for the two eta fields, which are
derived from those inputs and carry their shift. A fractional number with no tolerance declared
for it or a container above it is refused, so every fractional field has to declare one, and the
structure, the integers, the strings and the booleans still have to match exactly.

## API

| method | path                   | auth | description |
|--------|------------------------|------|-------------|
| POST   | `/predict`             | no   | `{distance_km, hour_of_day, day_of_week, pickup_zone_id, traffic_index, is_raining}` -> `{eta_minutes, model_version, request_id}`. Optional `X-Request-ID` header is echoed. |
| GET    | `/healthz`             | no   | Process liveness. |
| GET    | `/readyz`              | no   | 200 once a primary model is loaded, otherwise 503. |
| GET    | `/metrics`             | no   | Prometheus exposition. |
| GET    | `/admin/versions`      | token | Available versions, roles, load state, pool residency, swap history with `prewarmed` and `load_seconds`. |
| POST   | `/admin/shadow`        | token | `{version}` starts shadowing that version; `{version: null}` stops. |
| GET    | `/admin/shadow/report` | token | Divergence stats: count, mean/p50/p95/max abs delta, relative delta, share beyond threshold, bias. |
| GET    | `/admin/request-log`   | token | Whether capture is on, the path, sample rate, and records written. |
| POST   | `/admin/warm`          | token | `{version}` loads and warms a version into the pool; reports `load_seconds` and what was evicted. |
| POST   | `/admin/canary`        | token | `{version, weight}` routes `weight` (0..1] of traffic to that version; `{version: null}` clears. |
| GET    | `/admin/canary/report` | token | Canary status, weight, per-version samples, error rate, p50/p95 inference latency, thresholds, last rollback. |
| GET    | `/admin/drift`         | token | Per-feature drift score, status (`insufficient`, `stable`, `moderate`, `drifted`), unknown-category rate, live and training statistics. |
| POST   | `/admin/drift/reset`   | token | Clears the drift window. |
| POST   | `/admin/promote`       | token | `{version}` loads, warms, and swaps the primary. Clears the canary if it was that version. |
| POST   | `/admin/rollback`      | token | Swaps the primary back to the previous version. |

Admin endpoints require the `X-Admin-Token` header matching `MODELGATE_ADMIN_TOKEN`.
When the variable is unset the admin API answers 503.

Validation rules on `/predict` (any failure is a 422 with `{"error", "rejections": [{field, reason, message}]}`):

| field            | rule                                  | rejection reason |
|------------------|---------------------------------------|------------------|
| `distance_km`    | float, 0 to 500, finite               | `out_of_range`, `not_finite`, `wrong_type` |
| `hour_of_day`    | int, 0 to 23                          | `out_of_range`, `wrong_type` |
| `day_of_week`    | int, 0 to 6                           | `out_of_range`, `wrong_type` |
| `pickup_zone_id` | int in the manifest's zone set (1..12) | `unknown_zone`, `wrong_type` |
| `traffic_index`  | float, 0 to 1, finite                 | `out_of_range`, `not_finite`, `wrong_type` |
| `is_raining`     | bool (strict, `1` is not accepted)    | `wrong_type` |
| any other key    | rejected                              | `unknown_field` |
| missing key      | rejected                              | `missing_field` |
| invalid JSON     | rejected                              | `malformed_body` |

## Metrics

| metric | type | labels | meaning |
|--------|------|--------|---------|
| `modelgate_requests_total` | counter | `version`, `outcome` (`ok`, `rejected`, `error`) | Every `/predict` call. |
| `modelgate_request_latency_seconds` | histogram | `version` | End-to-end latency including validation and the shadow run. |
| `modelgate_predictions_eta_minutes` | histogram | `version` | Distribution of returned ETAs. |
| `modelgate_input_rejections_total` | counter | `reason` | Validation failures by reason. |
| `modelgate_shadow_divergence_minutes` | histogram | `primary`, `shadow` | abs(shadow ETA - primary ETA). |
| `modelgate_shadow_requests_total` | counter | `shadow`, `outcome` | Shadow inferences run. |
| `modelgate_model_version_info` | gauge | `version`, `role` | 1 for the version currently in that role. |
| `modelgate_version_swaps_total` | counter | `kind` (`promote`, `rollback`) | Primary swaps. |
| `modelgate_dropped_requests_total` | counter | | Requests that failed for a reason other than invalid input. Stays 0. |
| `modelgate_models_loaded` | gauge | | Versions resident in memory. |
| `modelgate_batch_size` | histogram | `version` | Rows per executed inference batch. |
| `modelgate_batch_queue_wait_seconds` | histogram | `version` | Time a request spent queued before its batch started. |
| `modelgate_batches_total` | counter | `version` | Inference batches executed. |
| `modelgate_swap_load_seconds` | histogram | `prewarmed` | Load and warm time before a promote could swap; 0 when the version was already resident. |
| `modelgate_warm_loads_total` | counter | `hit` | Explicit warm requests, by whether the version was already resident. |
| `modelgate_model_evictions_total` | counter | | Versions evicted from the pool. |
| `modelgate_model_pool_slots` | gauge | | Pool capacity. |
| `modelgate_request_log_records_total` | counter | | Accepted requests written to the sampled request log. |
| `modelgate_feature_drift` | gauge | `feature` | PSI of the live input window against training; 0 until `MODELGATE_DRIFT_MIN_SAMPLES` is reached. |
| `modelgate_feature_unknown_rate` | gauge | `feature` | Share of recent requests carrying a category not seen in training. |
| `modelgate_drift_window_samples` | gauge | | Accepted requests in the drift window. |
| `modelgate_canary_weight` | gauge | | Share of traffic routed to the canary; 0 when none is active. |
| `modelgate_canary_info` | gauge | `version` | 1 for the version serving as the canary. |
| `modelgate_canary_requests_total` | counter | `version`, `outcome` (`ok`, `fallback`) | Requests routed to the canary; `fallback` means the primary answered instead. |
| `modelgate_canary_rollbacks_total` | counter | `reason` (`error_rate`, `latency`) | Automatic canary rollbacks. |

## Grafana dashboard

`monitoring/grafana/dashboards/modelgate.json` is provisioned automatically by
`docker compose up`. Top row: stat tiles for the current primary and shadow version, the
dropped-request counter (green at 0, red otherwise), swap count, request rate, and p99
latency. Below that: stacked RPS by model version (the swap shows as v1 falling to zero and
v2 taking over in the same scrape), p50/p95/p99 latency, an ETA heatmap, input rejections
by reason as stacked bars, shadow divergence p50/p95 in minutes, shadow request rate with
swap events overlaid, dropped requests over time, and models loaded. Version swaps are also
drawn as annotations across every time series.

## Configuration

| variable | default | purpose |
|----------|---------|---------|
| `MODELGATE_ADMIN_TOKEN` | unset | Enables the admin API. |
| `MODELGATE_HOST` | `127.0.0.1` | `make serve` listener / Compose API publication address. A non-loopback value requires a non-default admin token. |
| `PORT` | `8000` | Native `make serve` port; Compose keeps port 8000. |
| `MODELGATE_ARTIFACTS_DIR` | `artifacts` | Where `manifest.json` and `eta_*.pt` live. |
| `MODELGATE_PRIMARY_VERSION` | lowest in manifest | Version loaded at startup. |
| `MODELGATE_SHADOW_VERSION` | unset | Shadow version loaded at startup. |
| `MODELGATE_SHADOW_THRESHOLD_MIN` | `2.0` | Divergence threshold for `share_beyond_threshold`. |
| `MODELGATE_SHADOW_LOG_SIZE` | `5000` | Rolling window of shadow records kept for the report. |
| `MODELGATE_CANARY_WINDOW_SECONDS` | `60` | Rolling window for per-version canary statistics. |
| `MODELGATE_CANARY_MIN_SAMPLES` | `50` | Samples required from both versions before a rollback verdict. |
| `MODELGATE_CANARY_ERROR_RATE_DELTA` | `0.02` | Roll back when the canary error rate exceeds the primary's by more than this. |
| `MODELGATE_CANARY_LATENCY_RATIO` | `2.0` | Roll back when the canary p95 inference latency exceeds the primary's by this factor. |
| `MODELGATE_CANARY_LATENCY_FLOOR_MS` | `1.0` | The p95 gap must also exceed this many milliseconds, so sub-millisecond noise does not trigger a rollback. |
| `MODELGATE_DRIFT_WINDOW_SIZE` | `2000` | Accepted requests kept per feature for drift scoring. |
| `MODELGATE_DRIFT_MIN_SAMPLES` | `100` | Samples before a feature gets a score and a status. |
| `MODELGATE_DRIFT_WARN_THRESHOLD` | `0.1` | PSI at or above this is `moderate`. |
| `MODELGATE_DRIFT_ALERT_THRESHOLD` | `0.25` | PSI at or above this is `drifted`. |
| `MODELGATE_DRIFT_REFRESH_EVERY` | `100` | Observations between gauge refreshes; the report always recomputes. |
| `MODELGATE_BATCH_MAX_SIZE` | `32` | Rows per inference batch and the padded row count of every forward pass. `1` disables batching. |
| `MODELGATE_BATCH_MAX_WAIT_MS` | `2.0` | Longest a dense-traffic request is held for others to join its batch. |
| `MODELGATE_MODEL_POOL_SIZE` | `3` | Versions kept resident; role holders are never evicted. |
| `MODELGATE_WARM_VERSIONS` | unset | Comma-separated versions to warm at startup. |
| `MODELGATE_REQUEST_LOG` | unset | JSON-lines path; setting it enables capture. |
| `MODELGATE_REQUEST_LOG_SAMPLE_RATE` | `0.1` | Share of accepted requests written, as an exact deterministic split. |

## Layout

```
modelgate/eval.py     replay harness behind `modelgate eval`
modelgate/model/      features.py, net.py, data.py, stats.py, train.py
modelgate/serving/    app.py, registry.py, batching.py, schemas.py, shadow.py, canary.py, drift.py,
                      reqlog.py, metrics.py, config.py
artifacts/            eta_v1.pt, eta_v2.pt, manifest.json (committed, reproducible)
loadtest/run.py       open-loop load generator with mid-run warm, canary, and promote
monitoring/           prometheus.yml, grafana provisioning and dashboard
tests/                validation, registry, shadow, canary, drift, batching, pool, eval and
                      request log, zero-drop swap, metrics, training; fixtures/replay_log.jsonl
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the swap mechanism, shadow design, and the
reasoning behind the validation and metrics choices.

## Releases

| version | adds |
|---------|------|
| 1.0.0 | Serving with strict input checks, shadow runs, atomic zero-drop version swaps, Prometheus metrics, Grafana dashboard, load test. |
| 2.0.0 | Canary routing: deterministic weighted split, per-version windows, automatic rollback on error rate or p95 latency, primary fallback so a broken canary drops nothing. |
| 3.0.0 | Feature drift: training statistics in the manifest, PSI per feature over a rolling window, unknown-category rate, `GET /admin/drift` and gauges. |
| 4.0.0 | Micro-batching with padded forward passes for bit-identical results, adaptive wait, FIFO fairness; warm model pool with `POST /admin/warm` and LRU eviction of role-free versions. |
| 5.0.0 | Sampled PII-free request log and `modelgate eval`: replay against two versions with MAE, calibration, divergence, and a logged-versus-replayed consistency check. |
| 5.0.1 | A suite that passes on any host: regenerated artifacts compared within a tolerance declared per field, wider canary and batching budgets; the browser demo on Vite 7.3.6 with 0 npm audit findings. |

Every release passed `ruff check`, `ruff format --check`, the full test suite, and the load
test with a mid-run swap at 0 dropped requests before it was tagged.

## Changelog

### 5.0.1

- Two tests compared a regenerated artifact with the committed one exactly, which does not
  hold across CPU architectures: regenerated from its seed on the x86-64 CI runner,
  `tests/fixtures/replay_log.jsonl` gets `distance_km` 8.5 on the row where the committed file
  records 8.499, and the `distance_km` standard deviation in `artifacts/manifest.json` comes
  out 4.990105 against the recorded 4.990104, while an arm64 Mac reproduces both exactly.
- Both now compare through `assert_reproduces` in `tests/conftest.py`, which holds each number
  to a tolerance declared per field: 0.001 for `distance_km` and 0.0001 for `traffic_index`,
  0.000001 for the training statistics, and 0.02 for the two eta fields derived from those
  inputs. Keys, field order, line count, integers, strings and booleans still have to match
  exactly, and a fractional number with no tolerance declared for it or a container above it
  is refused. `tests/test_artifact_tolerance.py` pins what the comparison admits and rejects,
  and `tests/measure_eta_shift.py` measures what the eta tolerance has to carry: one unit in
  the last place of each input moves the reference eta by at most 0.0094 minutes and the
  served prediction by at most 0.0028.
- `test_healthy_canary_stays_active`, whose healthy v2 canary was rolled back on the CI
  runner, raises `latency_floor_ms` to 250 ms, and `test_slow_canary_rolls_back_on_latency`
  still proves the rule with a 4 ms delay injected into the candidate. The batching test's
  `max_wait` rises from 5 ms to 50 ms with its assertions unchanged.
- The browser demo under `web/` moves from Vite 5.4.21 to 7.3.6 and `@vitejs/plugin-react`
  from 4.7.0 to 5.2.0 and declares node `^20.19.0 || >=22.12.0`. `npm audit` over
  `web/package-lock.json` reported esbuild (moderate, GHSA-67mh-4wv8-2f99) and Vite (high,
  three advisories including GHSA-fx2h-pf6j-xcff) at 5.0.0 and reports 0 vulnerabilities at
  5.0.1. A `web` CI job runs `npm ci`, the type-check, the 15 self-check assertions and the
  production bundle.
- 116 tests, passing on the x86-64 CI runner and on an arm64 Mac, and the load test with
  shadow, pre-warm, a 0.1 canary and a mid-run swap drops 0 of 1800 requests.

### 5.0.0

- `MODELGATE_REQUEST_LOG` turns on a sampled JSON-lines log of accepted requests: the six
  validated numeric inputs, the serving version, and the answer. No request id, header,
  or address is written. Sampling is an exact deterministic split.
- `modelgate eval --log FILE --versions A B` replays the log through the same validation,
  encoding, and padded forward pass the service uses and reports MAE, p95 error, bias,
  share within 2 minutes, and a calibration table with expected calibration error per
  version, the divergence between the two versions, and how many logged answers the
  replay reproduced. Two runs give the same report.
- `GET /admin/request-log` reports capture status; `modelgate_request_log_records_total`
  counts records. A 300-record fixture with the seeded actuals is committed for the tests.

### 4.0.0

- Dynamic micro-batching of concurrent `/predict` calls per model, FIFO, with a maximum
  batch size and maximum wait. Every forward pass is padded to the batch size, so results
  are bit-identical whether a row ran alone or in a full batch; the swap tests assert this
  across a promote under 50 workers.
- The wait is adaptive: a request that arrives more than the max wait after the previous
  one runs immediately, so sparse traffic keeps single-row latency (p50 1.7 ms at 200 rps
  on this machine, unchanged from 3.0.0).
- Warm pool: `POST /admin/warm` loads a version into a spare slot; the following promote
  reports `prewarmed: true` and `load_seconds: 0`. Versions with no role are evicted LRU
  when the pool is over capacity; the primary, previous, shadow, and canary never are.
- New metrics for batch size, queue wait, batches, swap load time, warm hits, and
  evictions. `loadtest.run --prewarm --burst N` exercises both.
- For this 19-input MLP the forward pass is about 0.2 ms of a 1.5 ms request, so batching
  changes throughput little; the mechanism matters for heavier models.

### 3.0.0

- Feature drift monitoring: the trainer records mean, std, quantiles, decile edges, and
  category frequencies per input in `manifest.json`; serving scores a rolling window of
  accepted inputs with a population stability index per feature.
- `GET /admin/drift` reports score, status, unknown-category rate (fed by `unknown_zone`
  rejections), and live versus training statistics; `POST /admin/drift/reset` clears the
  window.
- `modelgate_feature_drift{feature}`, `modelgate_feature_unknown_rate{feature}`, and
  `modelgate_drift_window_samples` gauges. Training-like traffic scores about 0.01 on
  every feature; tripling trip distance scores above 2 on `distance_km` alone.

### 2.0.0

- Canary routing: `POST /admin/canary {version, weight}` sends a deterministic weighted
  share of `/predict` traffic to a candidate version, served under its own version label
  in every metric.
- Automatic rollback: per-version error rate and p95 inference latency over a rolling
  window; the canary is cleared as soon as either exceeds the primary by the configured
  margin, and `GET /admin/canary/report` records why.
- A failing canary falls back to the primary answer for that request and counts against the
  candidate, so rollback happens with 0 dropped requests.
- `loadtest.run --canary-weight` enables a canary before the mid-run promote.

### 1.0.0

- `POST /predict` for the ETA MLP with strict input validation (types, ranges, known
  zones, finite floats, no unknown fields) and per-reason rejection counters.
- Model registry with a primary and an optional shadow version; shadow divergence is
  recorded on every request and summarised by `GET /admin/shadow/report`.
- Atomic version swaps: load and warm off the request path, swap under a lock, in-flight
  requests finish on the model they started with. 0 dropped requests in the load test.
- Prometheus metrics for traffic, latency, ETA distribution, rejections, shadow divergence,
  swaps, and drops, with a provisioned Grafana dashboard.
