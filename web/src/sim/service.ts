/**
 * The serving layer: port of the request path in modelgate/serving/app.py.
 *
 * A request is split into `admit` (validation, capture the primary) and
 * `finish` (inference, shadow run, metrics) so the load generator can hold
 * requests in flight across a swap exactly the way the async server does.
 */

import { encode, type Trip } from "./features";
import { MetricSet } from "./metrics";
import { Prng } from "./prng";
import { ModelRegistry, NoPrimaryError, type LoadedModel, type SwapRecord } from "./registry";
import { ShadowTracker, type ShadowReport } from "./shadow";
import { parseBody, validateBody, type Rejection } from "./validate";

export interface Settings {
  shadowThresholdMinutes: number;
  shadowLogSize: number;
  seed: number;
}

export const DEFAULT_SETTINGS: Settings = { shadowThresholdMinutes: 2.0, shadowLogSize: 5000, seed: 7 };

export interface PredictOk {
  status: 200;
  eta_minutes: number;
  model_version: string;
  request_id: string;
  latency_ms: number;
  shadow?: { version: string; eta_minutes: number; abs_delta: number };
}

export interface PredictRejected {
  status: 422;
  error: "invalid input";
  rejections: Rejection[];
  request_id: string;
}

export interface PredictError {
  status: 500 | 503;
  error: string;
  request_id: string;
}

export type PredictOutcome = PredictOk | PredictRejected | PredictError;

export interface InFlight {
  request_id: string;
  primary: LoadedModel;
  features: Float32Array;
  started: number; // performance.now() at admission
  sentAt: number; // sim seconds
}

export type Admitted = { kind: "inflight"; req: InFlight } | { kind: "done"; outcome: PredictOutcome };

export class Service {
  readonly metrics = new MetricSet();
  readonly registry: ModelRegistry;
  readonly tracker: ShadowTracker;
  readonly settings: Settings;
  private readonly ids: Prng;
  private simTime = 0;

  constructor(settings: Partial<Settings> = {}) {
    this.settings = { ...DEFAULT_SETTINGS, ...settings };
    this.registry = new ModelRegistry(this.metrics, () => this.simTime);
    this.tracker = new ShadowTracker(this.settings.shadowThresholdMinutes, this.settings.shadowLogSize);
    this.ids = new Prng(this.settings.seed * 7919 + 13);
  }

  /** Sim clock in seconds, advanced by whoever drives the service. */
  get now(): number {
    return this.simTime;
  }

  advance(dtSeconds: number): void {
    this.simTime += dtSeconds;
  }

  /** Lifespan: load the lowest version as primary. */
  boot(primary = "v1"): void {
    this.registry.promote(primary);
  }

  private nextRequestId(): string {
    let s = "";
    for (let i = 0; i < 4; i++) s += this.ids.int(0x10000).toString(16).padStart(4, "0");
    return s;
  }

  /**
   * Validation plus primary capture. Returns either an in-flight handle or a
   * finished outcome (422 for rejected input, 503 with no primary).
   */
  admit(body: unknown, requestId?: string): Admitted {
    const started = performance.now();
    const request_id = requestId ?? this.nextRequestId();
    const result = typeof body === "string" ? parseBody(body) : validateBody(body);
    if (!result.ok) {
      const reasons = new Set(result.rejections.map((r) => r.reason));
      for (const reason of reasons) this.metrics.inputRejections.labels({ reason }).inc();
      const primary = this.registry.primary;
      this.metrics.requests.labels({ version: primary ? primary.version : "none", outcome: "rejected" }).inc();
      return { kind: "done", outcome: { status: 422, error: "invalid input", rejections: result.rejections, request_id } };
    }
    let primary: LoadedModel;
    try {
      primary = this.registry.requirePrimary();
    } catch (err) {
      if (err instanceof NoPrimaryError) {
        this.metrics.droppedRequests.inc();
        this.metrics.requests.labels({ version: "none", outcome: "error" }).inc();
        return { kind: "done", outcome: { status: 503, error: "no primary model loaded", request_id } };
      }
      throw err;
    }
    return {
      kind: "inflight",
      req: { request_id, primary, features: encode(result.trip), started, sentAt: this.simTime },
    };
  }

  /** Inference on the captured primary, shadow run, metrics, response. */
  finish(req: InFlight): PredictOutcome {
    const { primary, features, request_id } = req;
    try {
      const eta = primary.predict(features);
      this.metrics.requests.labels({ version: primary.version, outcome: "ok" }).inc();
      this.metrics.predictionsEta.labels({ version: primary.version }).observe(eta);

      let shadowInfo: PredictOk["shadow"];
      const shadow = this.registry.shadow;
      if (shadow !== null && shadow.version !== primary.version) {
        shadowInfo = this.runShadow(shadow, primary, features, eta, request_id);
      }

      const latencyMs = performance.now() - req.started;
      this.metrics.requestLatency.labels({ version: primary.version }).observe(latencyMs / 1000);
      const out: PredictOk = {
        status: 200,
        eta_minutes: Math.round(eta * 100) / 100,
        model_version: primary.version,
        request_id,
        latency_ms: latencyMs,
      };
      if (shadowInfo) out.shadow = shadowInfo;
      return out;
    } catch (err) {
      // Unhandled exception on /predict: counted as dropped. Never happens in practice.
      this.metrics.droppedRequests.inc();
      this.metrics.requests.labels({ version: primary.version, outcome: "error" }).inc();
      return { status: 500, error: err instanceof Error ? err.message : "internal error", request_id };
    }
  }

  /** One-shot predict for the request builder. */
  predict(body: unknown): PredictOutcome {
    const admitted = this.admit(body);
    return admitted.kind === "done" ? admitted.outcome : this.finish(admitted.req);
  }

  private runShadow(
    shadow: LoadedModel,
    primary: LoadedModel,
    features: Float32Array,
    primaryEta: number,
    requestId: string,
  ): PredictOk["shadow"] {
    // The shadow path must never affect the client response.
    let shadowEta: number;
    try {
      shadowEta = shadow.predict(features);
    } catch {
      this.metrics.shadowRequests.labels({ shadow: shadow.version, outcome: "error" }).inc();
      this.tracker.recordError();
      return undefined;
    }
    const delta = Math.abs(shadowEta - primaryEta);
    this.metrics.shadowRequests.labels({ shadow: shadow.version, outcome: "ok" }).inc();
    this.metrics.shadowDivergence.labels({ primary: primary.version, shadow: shadow.version }).observe(delta);
    this.tracker.record({
      request_id: requestId,
      primary_version: primary.version,
      shadow_version: shadow.version,
      primary_eta: primaryEta,
      shadow_eta: shadowEta,
      at: this.simTime,
    });
    return { version: shadow.version, eta_minutes: Math.round(shadowEta * 100) / 100, abs_delta: delta };
  }

  // ---- admin ----------------------------------------------------------

  setShadow(version: string | null): string | null {
    if (version !== null && !this.registry.isKnown(version)) throw new Error(`unknown version ${version}`);
    if (version !== null && this.registry.primary?.version === version) {
      throw new Error("shadow version must differ from the primary");
    }
    const loaded = this.registry.setShadow(version);
    this.tracker.reset();
    return loaded ? loaded.version : null;
  }

  shadowReport(): ShadowReport & { primary: string | null; shadow: string | null } {
    return {
      ...this.tracker.report(),
      primary: this.registry.primary?.version ?? null,
      shadow: this.registry.shadow?.version ?? null,
    };
  }

  /**
   * Load and warm off the request path (a macrotask, standing in for
   * asyncio.to_thread), then swap under the lock.
   */
  promote(version: string): Promise<SwapRecord> {
    if (!this.registry.isKnown(version)) return Promise.reject(new Error(`unknown version ${version}`));
    return new Promise((resolve, reject) => {
      setTimeout(() => {
        try {
          this.registry.load(version);
          resolve(this.registry.promote(version));
        } catch (err) {
          reject(err);
        }
      }, 0);
    });
  }

  /** Synchronous variant for tests and the self-check. */
  promoteNow(version: string): SwapRecord {
    return this.registry.promote(version);
  }

  rollback(): SwapRecord {
    return this.registry.rollback();
  }

  exposition(): string {
    return this.metrics.expose();
  }

  trips(seed: number): TripSource {
    return new TripSource(seed);
  }
}

/** Random trips, the same distribution as loadtest/run.py::random_trip. */
export class TripSource {
  private readonly rng: Prng;

  constructor(seed: number) {
    this.rng = new Prng(seed);
  }

  next(): Trip {
    return {
      distance_km: Math.round(this.rng.logNormal(1.6, 0.7) * 100) / 100,
      hour_of_day: this.rng.int(24),
      day_of_week: this.rng.int(7),
      pickup_zone_id: this.rng.int(12) + 1,
      traffic_index: Math.round(this.rng.next() * 1000) / 1000,
      is_raining: this.rng.next() < 0.2,
    };
  }
}
