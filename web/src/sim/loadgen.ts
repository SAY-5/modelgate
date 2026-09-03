/**
 * Open-loop load generator: port of loadtest/run.py driven by a sim clock.
 *
 * Requests are admitted on schedule regardless of responses. Each one
 * captures the primary at admission and completes a short while later, so a
 * promote that lands between admission and completion is the exact case the
 * zero-drop guarantee covers. Anything that is not a 2xx is counted as
 * dropped, the same accounting as the real load test.
 */

import type { Trip } from "./features";
import { Prng } from "./prng";
import { Service, TripSource, type InFlight, type PredictOutcome } from "./service";

export interface Outcome {
  seq: number;
  sentAt: number;
  completedAt: number;
  latencyMs: number;
  status: number;
  version: string | null;
}

export type SecondSplit = { second: number; counts: Record<string, number>; dropped: number };

export interface LoadStats {
  sent: number;
  ok: number;
  dropped: number;
  rejected: number;
  achievedRps: number;
  elapsed: number;
  latency: { p50: number; p95: number; p99: number; max: number; mean: number };
}

interface Pending {
  seq: number;
  req: InFlight;
  completeAt: number;
}

const LATENCY_WINDOW = 4000;

export class LoadGen {
  rps: number;
  running = false;
  private carry = 0;
  private seq = 0;
  private pending: Pending[] = [];
  private readonly trips: TripSource;
  private readonly delay: Prng;
  private readonly latencies: number[] = [];
  private latencyHead = 0;
  private readonly sparkline: number[] = [];
  readonly timeline = new Map<number, SecondSplit>();
  private startedAt: number;
  private lastSentAt = 0;
  sent = 0;
  ok = 0;
  dropped = 0;
  rejected = 0;
  /** Fraction of admitted requests that are deliberately malformed (0 by default). */
  chaos = 0;
  private readonly chaosRng: Prng;
  lastOutcome: Outcome | null = null;

  constructor(
    readonly service: Service,
    opts: { rps?: number; seed?: number } = {},
  ) {
    this.rps = opts.rps ?? 200;
    const seed = opts.seed ?? 1;
    this.trips = service.trips(seed);
    this.delay = new Prng(seed * 31 + 7);
    this.chaosRng = new Prng(seed * 101 + 3);
    this.startedAt = service.now;
  }

  reset(): void {
    this.carry = 0;
    this.seq = 0;
    this.pending = [];
    this.latencies.length = 0;
    this.latencyHead = 0;
    this.sparkline.length = 0;
    this.timeline.clear();
    this.sent = this.ok = this.dropped = this.rejected = 0;
    this.startedAt = this.service.now;
    this.lastSentAt = 0;
    this.lastOutcome = null;
  }

  get elapsed(): number {
    return this.service.now - this.startedAt;
  }

  private malformed(trip: Trip): unknown {
    const pick = this.chaosRng.int(5);
    switch (pick) {
      case 0:
        return { ...trip, hour_of_day: String(trip.hour_of_day) };
      case 1:
        return { ...trip, pickup_zone_id: 13 };
      case 2:
        return { ...trip, distance_km: Number.NaN };
      case 3:
        return { ...trip, traffic_index: 1.4 };
      default:
        return { ...trip, driver_tip: 3 };
    }
  }

  /**
   * Advance the sim clock by dt seconds: admit the requests that fall due,
   * then complete the ones whose service time has elapsed.
   */
  tick(dt: number): Outcome[] {
    const completed: Outcome[] = [];
    if (this.running) {
      this.carry += this.rps * dt;
      const n = Math.floor(this.carry);
      this.carry -= n;
      const t0 = this.service.now;
      for (let i = 0; i < n; i++) {
        const sentAt = t0 + (dt * i) / Math.max(n, 1);
        this.admitOne(sentAt, completed);
      }
    }
    this.service.advance(dt);
    const now = this.service.now;
    if (this.pending.length) {
      const still: Pending[] = [];
      for (const p of this.pending) {
        if (p.completeAt <= now) completed.push(this.completeOne(p, now));
        else still.push(p);
      }
      this.pending = still;
    }
    return completed;
  }

  /** Complete every in-flight request immediately (used when stopping). */
  drain(): Outcome[] {
    const now = this.service.now;
    const out = this.pending.map((p) => this.completeOne(p, now));
    this.pending = [];
    return out;
  }

  get inFlight(): number {
    return this.pending.length;
  }

  private admitOne(sentAt: number, completed: Outcome[]): void {
    const seq = this.seq++;
    this.sent += 1;
    this.lastSentAt = sentAt;
    const trip = this.trips.next();
    const body = this.chaos > 0 && this.chaosRng.next() < this.chaos ? this.malformed(trip) : trip;
    const admitted = this.service.admit(body);
    if (admitted.kind === "done") {
      completed.push(this.record(seq, sentAt, sentAt, 0, admitted.outcome));
      return;
    }
    // Service time: a short log-normal delay stands in for the async hop.
    const serviceMs = this.delay.logNormal(Math.log(1.6), 0.45);
    this.pending.push({ seq, req: admitted.req, completeAt: sentAt + serviceMs / 1000 });
  }

  private completeOne(p: Pending, now: number): Outcome {
    const t0 = performance.now();
    const outcome = this.service.finish(p.req);
    const computeMs = performance.now() - t0;
    return this.record(p.seq, p.req.sentAt, now, computeMs, outcome);
  }

  private record(seq: number, sentAt: number, completedAt: number, computeMs: number, outcome: PredictOutcome): Outcome {
    const version = outcome.status === 200 ? outcome.model_version : null;
    const o: Outcome = { seq, sentAt, completedAt, latencyMs: computeMs, status: outcome.status, version };
    if (outcome.status === 200) {
      this.ok += 1;
      this.pushLatency(computeMs);
    } else if (outcome.status === 422) {
      this.rejected += 1;
    } else {
      this.dropped += 1;
    }
    const second = Math.floor(sentAt - this.startedAt);
    let split = this.timeline.get(second);
    if (!split) {
      split = { second, counts: {}, dropped: 0 };
      this.timeline.set(second, split);
      for (const key of [...this.timeline.keys()]) if (key < second - 60) this.timeline.delete(key);
    }
    if (version) split.counts[version] = (split.counts[version] ?? 0) + 1;
    if (outcome.status >= 500) split.dropped += 1;
    this.lastOutcome = o;
    return o;
  }

  private pushLatency(ms: number): void {
    if (this.latencies.length < LATENCY_WINDOW) this.latencies.push(ms);
    else {
      this.latencies[this.latencyHead] = ms;
      this.latencyHead = (this.latencyHead + 1) % LATENCY_WINDOW;
    }
  }

  /** Per-tick mean compute latency, for the sparkline. */
  pushSparkline(outcomes: Outcome[]): void {
    const ok = outcomes.filter((o) => o.status === 200);
    if (!ok.length) return;
    const mean = ok.reduce((a, o) => a + o.latencyMs, 0) / ok.length;
    this.sparkline.push(mean);
    if (this.sparkline.length > 160) this.sparkline.shift();
  }

  sparklineValues(): number[] {
    return this.sparkline;
  }

  stats(): LoadStats {
    const lat = [...this.latencies].sort((a, b) => a - b);
    const pct = (p: number) => (lat.length ? lat[Math.min(lat.length - 1, Math.max(0, Math.round((p / 100) * (lat.length - 1))))] : 0);
    const span = Math.max(this.lastSentAt - this.startedAt, 1e-9);
    return {
      sent: this.sent,
      ok: this.ok,
      dropped: this.dropped,
      rejected: this.rejected,
      achievedRps: this.sent > 1 ? this.sent / span : 0,
      elapsed: this.elapsed,
      latency: {
        p50: pct(50),
        p95: pct(95),
        p99: pct(99),
        max: lat.length ? lat[lat.length - 1] : 0,
        mean: lat.length ? lat.reduce((a, b) => a + b, 0) / lat.length : 0,
      },
    };
  }

  /** The last `n` seconds of version split, oldest first. */
  recentSplit(n: number): SecondSplit[] {
    const current = Math.floor(this.elapsed);
    const out: SecondSplit[] = [];
    for (let s = Math.max(0, current - n + 1); s <= current; s++) {
      out.push(this.timeline.get(s) ?? { second: s, counts: {}, dropped: 0 });
    }
    return out;
  }
}
