# Contributing

## Setup

```bash
uv sync --extra dev
make lint test
```

Use Python 3.12, [uv](https://docs.astral.sh/uv/) 0.11.7 (as pinned in CI), and Docker
Compose for the resolved-configuration tests. These tests do not need a running Docker daemon.
Torch is pulled from the CPU wheel index.

## Workflow

- Branch from `main`, keep commits single-line and in conventional style (`feat:`, `fix:`,
  `test:`, `docs:`, `chore:`).
- `make lint` and `make test` must pass before a pull request. CI runs the same commands
  plus a short load test with a mid-run version swap that fails on any dropped request.
- Model changes: run `make train` and commit the rebuilt `artifacts/` together with the
  code. Training is seeded, so the MAE in `artifacts/manifest.json` must match your run.
- New metrics go in `modelgate/serving/metrics.py` and get a panel in
  `monitoring/grafana/dashboards/modelgate.json` plus a row in the README metrics table.

## Testing notes

- Tests share the process-wide Prometheus registry. Assert on deltas, not absolute values.
- The zero-drop swap tests run 2000 requests across 50 workers. If you change the registry
  or the request path, run them a few times in a row (`pytest tests/test_swap_zero_drop.py`).
