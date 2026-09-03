# ModelGate

ML model serving and monitoring for a PyTorch ETA model. A FastAPI service with strict
input validation, shadow runs of candidate versions, Prometheus metrics on a provisioned
Grafana dashboard, and load-tested version swaps that complete with 0 dropped requests.

Python 3.12, PyTorch (CPU), FastAPI, prometheus_client, Grafana, uv.

## What it does

- Serves `POST /predict` for a small MLP that estimates trip ETA in minutes from distance,
  time of day, day of week, pickup zone, traffic index, and rain.
- Validates every input before a tensor is built: types, ranges, known zones, finite floats,
  no unknown fields. Rejections return 422 with a per-field reason and are counted by reason.
- Keeps a model registry with a primary and an optional shadow version. Shadow inference runs
  on every request, divergence against the primary is recorded, and the client always gets
  the primary answer.
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
                         |  ModelRegistry                                   |
                         |    primary ---> LoadedModel(v1)  --> response    |
                         |    shadow  ---> LoadedModel(v2)  --> divergence  |
                         |    previous     (rollback target)    log+metrics |
                         |        ^                                         |
  admin (X-Admin-Token)  |        | load + warm in thread, swap under lock   |
  POST /admin/promote -> |  /admin/shadow  /admin/rollback  /admin/versions |
                         |                                                  |
  Prometheus  <--------- |  GET /metrics      GET /healthz   GET /readyz    |
                         +--------------------------------------------------+
        |
        v
  Grafana (dashboard auto-provisioned from monitoring/grafana)

  artifacts/eta_v1.pt, eta_v2.pt, manifest.json   <--  modelgate.model.train (seeded)
```

## Quick start

```bash
uv sync --extra dev          # CPU torch from the PyTorch wheel index
make test                    # 55 tests including the zero-drop swap tests
make demo                    # start the server, 200 rps for 20 s, promote v2 at t+10 s
```

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

### Running the full stack

```bash
docker compose up --build
# service    http://localhost:8000/docs
# prometheus http://localhost:9090
# grafana    http://localhost:3000  (dashboard "ModelGate: ETA model serving", anonymous viewer)

# drive traffic against the container and swap versions mid-run
uv run python -m loadtest.run --url http://localhost:8000 --rps 200 --duration 60 --shadow-first
```

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

## API

| method | path                   | auth | description |
|--------|------------------------|------|-------------|
| POST   | `/predict`             | no   | `{distance_km, hour_of_day, day_of_week, pickup_zone_id, traffic_index, is_raining}` -> `{eta_minutes, model_version, request_id}`. Optional `X-Request-ID` header is echoed. |
| GET    | `/healthz`             | no   | Process liveness. |
| GET    | `/readyz`              | no   | 200 once a primary model is loaded, otherwise 503. |
| GET    | `/metrics`             | no   | Prometheus exposition. |
| GET    | `/admin/versions`      | token | Available versions, roles, load state, swap history. |
| POST   | `/admin/shadow`        | token | `{version}` starts shadowing that version; `{version: null}` stops. |
| GET    | `/admin/shadow/report` | token | Divergence stats: count, mean/p50/p95/max abs delta, relative delta, share beyond threshold, bias. |
| POST   | `/admin/promote`       | token | `{version}` loads, warms, and swaps the primary. |
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
| `MODELGATE_ARTIFACTS_DIR` | `artifacts` | Where `manifest.json` and `eta_*.pt` live. |
| `MODELGATE_PRIMARY_VERSION` | lowest in manifest | Version loaded at startup. |
| `MODELGATE_SHADOW_VERSION` | unset | Shadow version loaded at startup. |
| `MODELGATE_SHADOW_THRESHOLD_MIN` | `2.0` | Divergence threshold for `share_beyond_threshold`. |
| `MODELGATE_SHADOW_LOG_SIZE` | `5000` | Rolling window of shadow records kept for the report. |

## Layout

```
modelgate/model/      features.py, net.py, data.py, train.py
modelgate/serving/    app.py, registry.py, schemas.py, shadow.py, metrics.py, config.py
artifacts/            eta_v1.pt, eta_v2.pt, manifest.json (committed, reproducible)
loadtest/run.py       open-loop load generator with mid-run promote
monitoring/           prometheus.yml, grafana provisioning and dashboard
tests/                validation, registry, shadow, zero-drop swap, metrics, training
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the swap mechanism, shadow design, and the
reasoning behind the validation and metrics choices.

## Changelog

### 1.0.0

- `POST /predict` for the ETA MLP with strict input validation (types, ranges, known
  zones, finite floats, no unknown fields) and per-reason rejection counters.
- Model registry with a primary and an optional shadow version; shadow divergence is
  recorded on every request and summarised by `GET /admin/shadow/report`.
- Atomic version swaps: load and warm off the request path, swap under a lock, in-flight
  requests finish on the model they started with. 0 dropped requests in the load test.
- Prometheus metrics for traffic, latency, ETA distribution, rejections, shadow divergence,
  swaps, and drops, with a provisioned Grafana dashboard.
