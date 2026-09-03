"""Dynamic micro-batching of concurrent /predict calls.

Requests for the same loaded model are queued on the event loop and executed
in one forward pass. A batch runs as soon as it holds `max_batch_size` rows,
or `max_wait_s` after its first row arrived, whichever is first. Queues drain
in FIFO order, so the fairness bound is simple: a request waits at most
`max_wait_s` for its batch to start, plus one batch execution for every
`max_batch_size` requests that were queued ahead of it.

Every forward pass is padded to a fixed row count (see `LoadedModel`), so a
request gets bit-identical output whether it ran alone or in a full batch. A
`max_batch_size` of 1 disables batching: each call runs immediately.

The batcher is single-threaded by design. `submit` and the flush callbacks run
on the event loop, so no lock is needed and the ordering guarantee holds.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import torch

from modelgate.serving import metrics


@dataclass(frozen=True)
class BatchResult:
    eta: float
    queue_wait_s: float
    infer_s: float
    batch_size: int


@dataclass
class _Queue:
    model: object
    items: list[tuple[torch.Tensor, asyncio.Future, float]] = field(default_factory=list)
    timer: asyncio.TimerHandle | None = None  # max_wait deadline for the oldest row
    drain: asyncio.Handle | None = None  # flush already scheduled on the loop


@dataclass
class BatcherStats:
    batches: int = 0
    requests: int = 0
    max_batch: int = 0

    @property
    def mean_batch(self) -> float:
        return self.requests / self.batches if self.batches else 0.0


class Batcher:
    def __init__(self, max_batch_size: int = 32, max_wait_s: float = 0.002) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be at least 1")
        self.max_batch_size = max_batch_size
        self.max_wait_s = max(0.0, max_wait_s)
        self._queues: dict[int, _Queue] = {}
        self.stats = BatcherStats()

    async def submit(self, model, features: torch.Tensor) -> BatchResult:
        """Queue one feature row for `model` and wait for its result."""
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        key = id(model)
        queue = self._queues.get(key)
        if queue is None or queue.model is not model:
            queue = _Queue(model)
            self._queues[key] = queue
        queue.items.append((features, fut, time.perf_counter()))
        if len(queue.items) >= self.max_batch_size:
            # Full: run on the next loop step rather than inline, so every waiter in the
            # batch is parked on its future before results land and wakes in FIFO order.
            if queue.drain is None:
                queue.drain = loop.call_soon(self._flush, queue)
        elif queue.timer is None and queue.drain is None:
            queue.timer = loop.call_later(self.max_wait_s, self._flush, queue)
        return await fut

    def _flush(self, queue: _Queue) -> None:
        for handle in (queue.timer, queue.drain):
            if handle is not None:
                handle.cancel()
        queue.timer = queue.drain = None
        items = queue.items[: self.max_batch_size]
        del queue.items[: self.max_batch_size]
        if not items:
            return
        model = queue.model
        started = time.perf_counter()
        try:
            etas = model.predict_batch(torch.cat([row for row, _, _ in items]))
        except Exception as exc:  # noqa: BLE001
            for _, fut, _ in items:
                if not fut.done():
                    fut.set_exception(exc)
            self._reschedule(queue)
            return
        infer_s = time.perf_counter() - started
        size = len(items)
        self.stats.batches += 1
        self.stats.requests += size
        self.stats.max_batch = max(self.stats.max_batch, size)
        version = getattr(model, "version", "unknown")
        metrics.BATCH_SIZE.labels(version=version).observe(size)
        metrics.BATCHES.labels(version=version).inc()
        for (_, fut, queued_at), eta in zip(items, etas, strict=True):
            wait = started - queued_at
            metrics.BATCH_QUEUE_WAIT.labels(version=version).observe(wait)
            if not fut.done():
                fut.set_result(BatchResult(eta, wait, infer_s, size))
        self._reschedule(queue)

    def _reschedule(self, queue: _Queue) -> None:
        if queue.items:
            # Keep draining in FIFO order without waiting for the timer again.
            queue.drain = asyncio.get_running_loop().call_soon(self._flush, queue)
        elif self._queues.get(id(queue.model)) is queue:
            del self._queues[id(queue.model)]

    @property
    def pending(self) -> int:
        return sum(len(q.items) for q in self._queues.values())
