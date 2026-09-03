"""Open-loop load generator that swaps model versions mid-run.

Drives POST /predict at a fixed request rate, optionally pre-warms the
candidate at 15% of the run, enables a shadow run at 25% and a weighted canary
at 35%, promotes the candidate version at 50%, and accounts for every single
request. A request is "dropped" if it got any non-2xx response
or no response at all (connection error, timeout). The point of the exercise
is that the dropped count is 0 across the swap.

    python -m loadtest.run --spawn-server --rps 200 --duration 20 --shadow-first
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import httpx

ZONES = list(range(1, 13))


@dataclass
class Outcome:
    seq: int
    sent_at: float  # seconds since run start
    latency_ms: float | None
    status: int | None
    version: str | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status is not None and 200 <= self.status < 300


@dataclass
class RunResult:
    outcomes: list[Outcome] = field(default_factory=list)
    started_wall: float = 0.0
    warm_at: float | None = None
    warm_record: dict | None = None
    shadow_at: float | None = None
    canary_at: float | None = None
    swap_requested_at: float | None = None
    swap_done_at: float | None = None
    swap_record: dict | None = None
    server_dropped_metric: float | None = None
    shadow_report: dict | None = None
    canary_report: dict | None = None
    batch_stats: dict | None = None


def random_trip(rng: random.Random) -> dict:
    return {
        "distance_km": round(rng.lognormvariate(1.6, 0.7), 2),
        "hour_of_day": rng.randrange(24),
        "day_of_week": rng.randrange(7),
        "pickup_zone_id": rng.choice(ZONES),
        "traffic_index": round(rng.random(), 3),
        "is_raining": rng.random() < 0.2,
    }


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return ordered[idx]


async def wait_ready(client: httpx.AsyncClient, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            r = await client.get("/readyz")
            if r.status_code == 200:
                return
            last = r.text
        except httpx.HTTPError as exc:
            last = repr(exc)
        await asyncio.sleep(0.2)
    raise SystemExit(f"server not ready after {timeout}s: {last}")


async def run_load(args: argparse.Namespace) -> RunResult:
    result = RunResult()
    rng = random.Random(args.seed)
    admin = {"X-Admin-Token": args.admin_token}
    limits = httpx.Limits(max_connections=args.max_connections, max_keepalive_connections=256)
    timeout = httpx.Timeout(args.request_timeout)
    async with httpx.AsyncClient(base_url=args.url, limits=limits, timeout=timeout) as client:
        await wait_ready(client, args.ready_timeout)
        await client.post("/admin/shadow", json={"version": None}, headers=admin)
        r = await client.post("/admin/promote", json={"version": args.from_version}, headers=admin)
        r.raise_for_status()

        start = time.perf_counter()
        result.started_wall = time.time()
        interval = 1.0 / args.rps
        total = int(args.rps * args.duration)
        tasks: list[asyncio.Task] = []

        async def fire(seq: int, planned_at: float) -> None:
            payload = random_trip(rng)
            t0 = time.perf_counter()
            try:
                resp = await client.post("/predict", json=payload)
                latency = (time.perf_counter() - t0) * 1000
                version = resp.json().get("model_version") if resp.status_code == 200 else None
                result.outcomes.append(Outcome(seq, planned_at, latency, resp.status_code, version))
            except httpx.HTTPError as exc:
                result.outcomes.append(Outcome(seq, planned_at, None, None, None, repr(exc)))

        async def control() -> None:
            if args.prewarm:
                await asyncio.sleep(args.duration * 0.15)
                result.warm_at = time.perf_counter() - start
                r = await client.post(
                    "/admin/warm", json={"version": args.promote_to}, headers=admin
                )
                r.raise_for_status()
                result.warm_record = r.json()
            if args.shadow_first:
                await asyncio.sleep(args.duration * 0.25)
                result.shadow_at = time.perf_counter() - start
                r = await client.post(
                    "/admin/shadow", json={"version": args.promote_to}, headers=admin
                )
                r.raise_for_status()
            if args.canary_weight > 0:
                await asyncio.sleep(max(0.0, args.duration * 0.35 - (time.perf_counter() - start)))
                result.canary_at = time.perf_counter() - start
                r = await client.post(
                    "/admin/canary",
                    json={"version": args.promote_to, "weight": args.canary_weight},
                    headers=admin,
                )
                r.raise_for_status()
            await asyncio.sleep(max(0.0, args.duration * 0.5 - (time.perf_counter() - start)))
            result.swap_requested_at = time.perf_counter() - start
            r = await client.post(
                "/admin/promote", json={"version": args.promote_to}, headers=admin
            )
            r.raise_for_status()
            result.swap_done_at = time.perf_counter() - start
            result.swap_record = r.json()

        controller = asyncio.create_task(control())
        burst = max(1, args.burst)
        for seq in range(total):
            # With --burst N, N requests are released together every N intervals; the
            # average rate is unchanged and concurrent arrivals exercise the batcher.
            planned = (seq // burst) * burst * interval
            now = time.perf_counter() - start
            if planned > now:
                await asyncio.sleep(planned - now)
            tasks.append(asyncio.create_task(fire(seq, time.perf_counter() - start)))
        await asyncio.gather(*tasks)
        await controller

        if args.shadow_first:
            r = await client.get("/admin/shadow/report", headers=admin)
            if r.status_code == 200:
                result.shadow_report = r.json()
        if args.canary_weight > 0:
            r = await client.get("/admin/canary/report", headers=admin)
            if r.status_code == 200:
                result.canary_report = r.json()
        metrics_text = (await client.get("/metrics")).text
        sums: dict[str, float] = {}
        for line in metrics_text.splitlines():
            if line.startswith("modelgate_dropped_requests_total "):
                result.server_dropped_metric = float(line.split()[-1])
            for name in ("modelgate_batch_size", "modelgate_batch_queue_wait_seconds"):
                for suffix in ("_sum", "_count"):
                    if line.startswith(name + suffix):
                        sums[name + suffix] = sums.get(name + suffix, 0.0) + float(line.split()[-1])
        batches = sums.get("modelgate_batch_size_count", 0.0)
        if batches:
            result.batch_stats = {
                "batches": int(batches),
                "mean_batch_size": round(sums["modelgate_batch_size_sum"] / batches, 2),
                "mean_queue_wait_ms": round(
                    1000
                    * sums["modelgate_batch_queue_wait_seconds_sum"]
                    / max(sums.get("modelgate_batch_queue_wait_seconds_count", 1.0), 1.0),
                    3,
                ),
            }
    return result


def summarize(result: RunResult, args: argparse.Namespace) -> dict:
    outcomes = sorted(result.outcomes, key=lambda o: o.seq)
    ok = [o for o in outcomes if o.ok]
    dropped = [o for o in outcomes if not o.ok]
    latencies = [o.latency_ms for o in ok if o.latency_ms is not None]
    span = max((o.sent_at for o in outcomes), default=0.0) or 1.0
    swap_t = result.swap_done_at
    before = Counter(o.version for o in ok if swap_t is None or o.sent_at < swap_t)
    after = Counter(o.version for o in ok if swap_t is not None and o.sent_at >= swap_t)

    timeline: dict[int, Counter] = {}
    for o in ok:
        timeline.setdefault(int(o.sent_at), Counter())[o.version] += 1

    return {
        "target_rps": args.rps,
        "duration_s": args.duration,
        "total_requests": len(outcomes),
        "successes": len(ok),
        "dropped": len(dropped),
        "dropped_detail": Counter(
            (o.status if o.status is not None else "no_response") for o in dropped
        ),
        "server_dropped_metric": result.server_dropped_metric,
        "achieved_rps": round(len(outcomes) / span, 1),
        "latency_ms": {
            "p50": round(percentile(latencies, 50), 2),
            "p95": round(percentile(latencies, 95), 2),
            "p99": round(percentile(latencies, 99), 2),
            "max": round(max(latencies), 2) if latencies else 0.0,
            "mean": round(statistics.fmean(latencies), 2) if latencies else 0.0,
        },
        "warm_at_s": round(result.warm_at, 3) if result.warm_at else None,
        "warm_record": result.warm_record,
        "batching": result.batch_stats,
        "shadow_enabled_at_s": round(result.shadow_at, 3) if result.shadow_at else None,
        "canary_enabled_at_s": round(result.canary_at, 3) if result.canary_at else None,
        "canary_weight": args.canary_weight,
        "canary_report": result.canary_report,
        "swap": {
            "from": args.from_version,
            "to": args.promote_to,
            "requested_at_s": round(result.swap_requested_at, 3)
            if result.swap_requested_at
            else None,
            "completed_at_s": round(result.swap_done_at, 3) if result.swap_done_at else None,
            "wall_clock": (
                time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(result.started_wall + result.swap_done_at)
                )
                if result.swap_done_at
                else None
            ),
            "server_record": result.swap_record,
        },
        "versions_before_swap": dict(before),
        "versions_after_swap": dict(after),
        "timeline": {str(s): dict(c) for s, c in sorted(timeline.items())},
        "shadow_report": result.shadow_report,
    }


def print_summary(s: dict) -> None:
    line = "=" * 64
    print(line)
    print("ModelGate load test: version swap under load")
    print(line)
    print(f"target rate        {s['target_rps']} rps for {s['duration_s']} s")
    print(f"achieved rate      {s['achieved_rps']} rps")
    print(f"total requests     {s['total_requests']}")
    print(f"successes (2xx)    {s['successes']}")
    print(f"dropped requests   {s['dropped']}  (non-2xx or no response)")
    if s["dropped"]:
        print(f"  breakdown        {dict(s['dropped_detail'])}")
    print(f"server dropped ctr {s['server_dropped_metric']}  (modelgate_dropped_requests_total)")
    lat = s["latency_ms"]
    print(
        f"latency ms         p50 {lat['p50']}  p95 {lat['p95']}  p99 {lat['p99']}  max {lat['max']}"
    )
    if s["batching"]:
        b = s["batching"]
        print(
            f"batching           {b['batches']} batches, mean size {b['mean_batch_size']}, "
            f"mean queue wait {b['mean_queue_wait_ms']} ms"
        )
    if s["warm_at_s"] is not None:
        w = s["warm_record"] or {}
        print(
            f"warm pool          t+{s['warm_at_s']}s ({s['swap']['to']} loaded in "
            f"{w.get('load_seconds')}s, resident {w.get('resident')})"
        )
    if s["shadow_enabled_at_s"] is not None:
        print(f"shadow enabled     t+{s['shadow_enabled_at_s']}s ({s['swap']['to']} shadowing)")
    if s["canary_enabled_at_s"] is not None:
        print(
            f"canary enabled     t+{s['canary_enabled_at_s']}s "
            f"({s['swap']['to']} at weight {s['canary_weight']})"
        )
    sw = s["swap"]
    rec = sw["server_record"] or {}
    warm_note = ""
    if "prewarmed" in rec:
        warm_note = f", prewarmed={rec['prewarmed']} load={rec.get('load_seconds')}s"
    print(
        f"swap {sw['from']} -> {sw['to']}     requested t+{sw['requested_at_s']}s, "
        f"completed t+{sw['completed_at_s']}s ({sw['wall_clock']}{warm_note})"
    )
    print(f"versions before    {s['versions_before_swap']}")
    print(f"versions after     {s['versions_after_swap']}")
    print("per-second split   " + "  ".join(f"{k}s:{v}" for k, v in s["timeline"].items()))
    if s["shadow_report"]:
        rep = s["shadow_report"]
        print(
            f"shadow report      n={rep['count']} mean|d|={rep['abs_delta_minutes']['mean']} "
            f"p95|d|={rep['abs_delta_minutes']['p95']} "
            f"beyond {rep['threshold_minutes']}min={rep['share_beyond_threshold']}"
        )
    if s["canary_report"]:
        rep = s["canary_report"]
        cand = rep["versions"].get(s["swap"]["to"], {})
        print(
            f"canary report      status={rep['status']} rollbacks={rep['rollbacks']} "
            f"candidate samples={cand.get('samples', 0)} "
            f"errors={cand.get('errors', 0)} p95={cand.get('p95_ms', 0.0)}ms"
        )
    print(line)
    verdict = "PASS: 0 dropped requests across the swap" if s["dropped"] == 0 else "FAIL"
    print(verdict)
    print(line)


def spawn_server(port: int, admin_token: str, log_path: Path) -> subprocess.Popen:
    env = {**os.environ, "MODELGATE_ADMIN_TOKEN": admin_token}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "w")  # noqa: SIM115
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "modelgate.serving.app:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-level",
        "warning",
    ]
    return subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--url", default=None, help="base URL of a running server")
    p.add_argument("--spawn-server", action="store_true", help="start uvicorn locally")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--rps", type=float, default=200)
    p.add_argument("--duration", type=float, default=20)
    p.add_argument("--from-version", default="v1")
    p.add_argument("--promote-to", default="v2")
    p.add_argument("--shadow-first", action="store_true")
    p.add_argument("--prewarm", action="store_true", help="warm the candidate before the swap")
    p.add_argument("--burst", type=int, default=1, help="release requests in groups of N")
    p.add_argument(
        "--canary-weight",
        type=float,
        default=0.0,
        help="route this share of traffic to the candidate before promoting it (0 = off)",
    )
    p.add_argument("--admin-token", default=os.environ.get("MODELGATE_ADMIN_TOKEN", "dev-token"))
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-connections", type=int, default=512)
    p.add_argument("--request-timeout", type=float, default=5.0)
    p.add_argument("--ready-timeout", type=float, default=60.0)
    p.add_argument("--json-out", type=Path, default=Path("loadtest/results/last.json"))
    p.add_argument("--fail-on-drops", action="store_true")
    args = p.parse_args(argv)
    if args.url is None:
        args.url = f"http://127.0.0.1:{args.port}"
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    server = None
    if args.spawn_server:
        server = spawn_server(args.port, args.admin_token, Path("loadtest/results/server.log"))
    try:
        result = asyncio.run(run_load(args))
    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
    summary = summarize(result, args)
    print_summary(summary)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    if args.fail_on_drops and summary["dropped"] != 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
