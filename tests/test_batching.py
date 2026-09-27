"""Micro-batching: exact per-request results, bounded wait, FIFO, zero-drop swap."""

from __future__ import annotations

import asyncio
import time
from collections import Counter

import httpx
import pytest
import torch

from modelgate.model.data import make_dataset
from modelgate.serving import metrics
from modelgate.serving.app import create_app
from modelgate.serving.batching import Batcher
from modelgate.serving.registry import ModelRegistry
from tests.conftest import ADMIN, ARTIFACTS, make_settings


def _rows(n: int, seed: int) -> list[dict]:
    return make_dataset(n, seed)[2]


def _features(n: int, seed: int) -> torch.Tensor:
    return make_dataset(n, seed)[0]


def test_padded_batches_match_single_rows_bit_for_bit():
    reg = ModelRegistry(ARTIFACTS, pad_rows=32)
    x = _features(600, seed=31)
    for version in ("v1", "v2"):
        model = reg.load(version)
        alone = [model.predict(x[i : i + 1]) for i in range(len(x))]
        g = torch.Generator().manual_seed(2)
        for _ in range(60):
            size = int(torch.randint(1, 65, (1,), generator=g))
            idx = torch.randperm(len(x), generator=g)[:size]
            batched = model.predict_batch(x[idx])
            assert batched == [alone[int(i)] for i in idx]


async def test_batcher_groups_concurrent_submits_and_preserves_results():
    reg = ModelRegistry(ARTIFACTS, pad_rows=16)
    model = reg.load("v1")
    batcher = Batcher(max_batch_size=16, max_wait_s=0.005)
    x = _features(64, seed=32)
    expected = [model.predict(x[i : i + 1]) for i in range(64)]
    batches_before = metrics.counter_value(metrics.BATCHES, version="v1")

    results = await asyncio.gather(*(batcher.submit(model, x[i : i + 1]) for i in range(64)))
    assert [r.eta for r in results] == expected
    assert batcher.stats.batches == 4 and batcher.stats.max_batch == 16
    assert batcher.stats.requests == 64 and batcher.stats.mean_batch == 16.0
    assert metrics.counter_value(metrics.BATCHES, version="v1") == batches_before + 4
    assert {r.batch_size for r in results} == {16}
    assert batcher.pending == 0


async def test_sparse_requests_do_not_wait_and_dense_ones_wait_at_most_max_wait():
    reg = ModelRegistry(ARTIFACTS, pad_rows=8)
    model = reg.load("v1")
    # 50ms rather than 5: the assertions below are wall clock around a forward pass on an event
    # loop, and a shared runner's scheduling jitter alone can cost several milliseconds. The
    # relationships being tested are unchanged; only the budget is large enough to be about the
    # batcher rather than about the machine.
    max_wait = 0.05
    batcher = Batcher(max_batch_size=8, max_wait_s=max_wait)
    x = _features(4, seed=33)
    # Sparse: each request arrives well after the previous one, so it runs at once.
    for _ in range(10):
        await asyncio.sleep(max_wait * 3)
        t0 = time.perf_counter()
        result = await batcher.submit(model, x[:1])
        assert result.batch_size == 1
        assert result.queue_wait_s < max_wait
        assert time.perf_counter() - t0 < max_wait
    # Dense: a second request within max_wait of the first is held for up to max_wait
    # so a third can join it; nothing waits longer than max_wait plus scheduling slack.
    await asyncio.sleep(max_wait * 3)
    first = asyncio.create_task(batcher.submit(model, x[:1]))
    await asyncio.sleep(max_wait / 5)
    second = asyncio.create_task(batcher.submit(model, x[1:2]))
    await asyncio.sleep(max_wait / 5)
    third = asyncio.create_task(batcher.submit(model, x[2:3]))
    results = await asyncio.gather(first, second, third)
    assert results[0].batch_size == 1
    assert results[1].batch_size == 2 and results[2].batch_size == 2
    assert max_wait * 0.5 <= results[1].queue_wait_s < max_wait + 0.05
    assert results[2].queue_wait_s < max_wait + 0.05


async def test_full_batch_does_not_wait_for_the_timer():
    reg = ModelRegistry(ARTIFACTS, pad_rows=4)
    model = reg.load("v1")
    batcher = Batcher(max_batch_size=4, max_wait_s=1.0)
    x = _features(4, seed=34)
    t0 = time.perf_counter()
    results = await asyncio.gather(*(batcher.submit(model, x[i : i + 1]) for i in range(4)))
    assert time.perf_counter() - t0 < 0.5
    assert all(r.queue_wait_s < 0.05 for r in results)


async def test_queue_drains_in_fifo_order():
    reg = ModelRegistry(ARTIFACTS, pad_rows=8)
    model = reg.load("v1")
    batcher = Batcher(max_batch_size=8, max_wait_s=0.002)
    x = _features(100, seed=35)
    order: list[int] = []

    async def one(i: int) -> None:
        await batcher.submit(model, x[i : i + 1])
        order.append(i)

    await asyncio.gather(*(one(i) for i in range(100)))
    assert order == list(range(100))
    # 100 rows over batches of 8: the last one ran with 4 rows.
    assert batcher.stats.batches == 13 and batcher.stats.max_batch == 8


async def test_batch_failure_reaches_every_waiter_and_queue_recovers():
    reg = ModelRegistry(ARTIFACTS, pad_rows=8)
    good = reg.load("v1")

    class Broken:
        version = "broken"

        def predict_batch(self, _):
            raise RuntimeError("boom")

    batcher = Batcher(max_batch_size=8, max_wait_s=0.002)
    x = _features(6, seed=36)
    results = await asyncio.gather(
        *(batcher.submit(Broken(), x[i : i + 1]) for i in range(6)), return_exceptions=True
    )
    assert all(isinstance(r, RuntimeError) for r in results)
    assert (await batcher.submit(good, x[:1])).eta == good.predict(x[:1])
    assert batcher.pending == 0


def test_batch_size_one_disables_batching():
    batcher = Batcher(max_batch_size=1, max_wait_s=1.0)
    reg = ModelRegistry(ARTIFACTS, pad_rows=1)
    model = reg.load("v1")
    x = _features(1, seed=37)

    async def run():
        t0 = time.perf_counter()
        result = await batcher.submit(model, x)
        assert time.perf_counter() - t0 < 0.5 and result.batch_size == 1

    asyncio.run(run())
    with pytest.raises(ValueError):
        Batcher(max_batch_size=0)


async def test_predict_results_identical_under_concurrent_batching():
    app = create_app(make_settings(batch_max_size=32, batch_max_wait_ms=2.0))
    rows = _rows(400, seed=38)
    dropped = metrics.counter_value(metrics.DROPPED_REQUESTS)
    async with app.router.lifespan_context(app):
        primary = app.state.registry.primary
        x = _features(400, seed=38)
        expected = [round(primary.predict(x[i : i + 1]), 2) for i in range(400)]
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:

            async def one(i: int):
                await asyncio.sleep(0)
                return await client.post("/predict", json=rows[i])

            responses = await asyncio.gather(*(one(i) for i in range(400)))
    assert Counter(r.status_code for r in responses) == {200: 400}
    assert [r.json()["eta_minutes"] for r in responses] == expected
    assert metrics.counter_value(metrics.DROPPED_REQUESTS) == dropped
    stats = app.state.batcher.stats
    assert stats.requests == 400 and stats.batches < 400 and stats.max_batch > 1

    text = (await _get_metrics(app)).text
    assert 'modelgate_batches_total{version="v1"}' in text
    assert 'modelgate_batch_size_bucket{le="32.0",version="v1"}' in text
    assert 'modelgate_batch_queue_wait_seconds_count{version="v1"}' in text


async def _get_metrics(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        return await client.get("/metrics")


async def test_swap_under_batching_drops_nothing_and_stays_exact():
    app = create_app(make_settings(batch_max_size=32, batch_max_wait_ms=2.0))
    total, workers = 2000, 50
    rows = _rows(total, seed=39)
    x = _features(total, seed=39)
    dropped = metrics.counter_value(metrics.DROPPED_REQUESTS)
    outcomes: list[tuple[int, str | None, float | None]] = []
    done = 0

    async with app.router.lifespan_context(app):
        registry = app.state.registry
        expected = {
            v: [round(registry.load(v).predict(x[i : i + 1]), 2) for i in range(total)]
            for v in ("v1", "v2")
        }
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            sem = asyncio.Semaphore(workers)

            async def one(i: int) -> None:
                nonlocal done
                async with sem:
                    await asyncio.sleep(0)
                    r = await client.post(
                        "/predict", json=rows[i], headers={"X-Request-Id": str(i)}
                    )
                body = r.json() if r.status_code == 200 else {}
                outcomes.append(
                    (r.status_code, body.get("model_version"), body.get("eta_minutes"), i)
                )
                done += 1

            async def promoter() -> None:
                while done < total // 2:
                    await asyncio.sleep(0.001)
                r = await client.post("/admin/promote", json={"version": "v2"}, headers=ADMIN)
                assert r.status_code == 200 and r.json()["to"] == "v2"

            await asyncio.gather(promoter(), *(one(i) for i in range(total)))

    assert Counter(s for s, _, _, _ in outcomes) == {200: total}
    assert metrics.counter_value(metrics.DROPPED_REQUESTS) == dropped
    versions = Counter(v for _, v, _, _ in outcomes)
    assert versions["v1"] > 0 and versions["v2"] > 0
    # Every answer equals the single-row answer of the version that served it, on both
    # sides of the swap and regardless of which batch the row landed in.
    assert all(eta == expected[v][i] for _, v, eta, i in outcomes)
    assert app.state.batcher.stats.max_batch > 1
