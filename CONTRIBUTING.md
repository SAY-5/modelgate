# Contributing

## Setup

```bash
uv sync --extra dev
make lint test
```

Python 3.12 and [uv](https://docs.astral.sh/uv/) are the only prerequisites. Torch is
pulled from the CPU wheel index, so the environment installs in under a minute.

## Workflow

- Branch from `main`, keep commits single-line and in conventional style (`feat:`, `fix:`,
  `test:`, `docs:`, `chore:`).
- `make lint` and `make test` must pass before a pull request. CI runs the same commands
  plus a short load test with a mid-run version swap that fails on any dropped request.
- Model changes: run `make train` and commit the regenerated `artifacts/` together with the
  code. Training is seeded, so the MAE in `artifacts/manifest.json` must match your run.
- New metrics go in `modelgate/serving/metrics.py` and get a panel in
  `monitoring/grafana/dashboards/modelgate.json` plus a row in the README metrics table.

## Testing notes

- Tests share the process-wide Prometheus registry. Assert on deltas, not absolute values.
- The zero-drop swap tests run 2000 requests across 50 workers. If you change the registry
  or the request path, run them a few times in a row (`pytest tests/test_swap_zero_drop.py`).
