.PHONY: install lint test train serve loadtest demo clean

UV ?= uv
PORT ?= 8000
export PORT
DURATION ?= 20
RPS ?= 200

install:
	$(UV) sync --extra dev

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .

fmt:
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

test:
	$(UV) run pytest -q

train:
	$(UV) run python -m modelgate.model.train

artifacts/manifest.json:
	$(UV) run python -m modelgate.model.train

serve:
	$(UV) run python -m modelgate.serving.startup

loadtest: artifacts/manifest.json
	$(UV) run python -m loadtest.run --spawn-server --port $(PORT) --rps $(RPS) --duration $(DURATION) --shadow-first

demo: loadtest

clean:
	rm -rf .pytest_cache .ruff_cache loadtest/results
