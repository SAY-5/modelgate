/**
 * Model registry with atomic, zero-drop version swaps.
 * Port of modelgate/serving/registry.py.
 *
 * A version is built from its weights and warmed with a dummy forward pass
 * before it can be selected. The swap itself is a single reference
 * assignment. Every request captures its own reference to the primary at the
 * start, so a request in flight during a swap finishes on the model it
 * started with.
 */

import { FEATURE_DIM } from "./features";
import { MANIFEST } from "./manifest";
import type { MetricSet } from "./metrics";
import { EtaNet, WEIGHTS } from "./net";

export class UnknownVersionError extends Error {}
export class NoPrimaryError extends Error {}

export interface LoadedModel {
  readonly version: string;
  readonly net: EtaNet;
  readonly testMae: number | null;
  readonly loadedAt: number; // sim seconds
  readonly loadMs: number; // measured build + warm time
  predict(features: Float32Array): number;
}

export interface SwapRecord {
  kind: "promote" | "rollback";
  from: string | null;
  to: string;
  changed: boolean;
  at: number; // sim seconds
  swap_micros: number; // measured duration of the critical section
  load_ms: number; // measured load + warm time, 0 when already resident
}

export class ModelRegistry {
  private readonly loaded = new Map<string, LoadedModel>();
  private _primary: LoadedModel | null = null;
  private _previous: LoadedModel | null = null;
  private _shadow: LoadedModel | null = null;
  readonly swapHistory: SwapRecord[] = [];

  constructor(
    private readonly metrics: MetricSet,
    private readonly now: () => number,
  ) {}

  availableVersions(): string[] {
    return Object.keys(MANIFEST.versions).sort();
  }

  isKnown(version: string): boolean {
    return version in MANIFEST.versions;
  }

  isLoaded(version: string): boolean {
    return this.loaded.has(version);
  }

  /** Build and warm a version. Idempotent. */
  load(version: string): LoadedModel {
    if (!this.isKnown(version)) throw new UnknownVersionError(version);
    const cached = this.loaded.get(version);
    if (cached) return cached;
    const t0 = performance.now();
    const net = new EtaNet(WEIGHTS[version]);
    const entry = MANIFEST.versions[version];
    const model: LoadedModel = {
      version,
      net,
      testMae: entry.metrics.test_mae_minutes,
      loadedAt: this.now(),
      loadMs: 0,
      predict: (features) => net.forward(features),
    };
    // Warm-up: the first forward pass pays for any lazy work, so take it here.
    model.predict(new Float32Array(FEATURE_DIM));
    (model as { loadMs: number }).loadMs = performance.now() - t0;
    this.loaded.set(version, model);
    this.metrics.modelsLoaded.set(this.loaded.size);
    return model;
  }

  get primary(): LoadedModel | null {
    return this._primary;
  }

  get shadow(): LoadedModel | null {
    return this._shadow;
  }

  get previous(): LoadedModel | null {
    return this._previous;
  }

  requirePrimary(): LoadedModel {
    const p = this._primary;
    if (p === null) throw new NoPrimaryError("no primary model loaded");
    return p;
  }

  get ready(): boolean {
    return this._primary !== null;
  }

  /** Make `version` the primary. Loads and warms first, then swaps atomically. */
  promote(version: string): SwapRecord {
    const candidate = this.load(version);
    const t0 = performance.now();
    const old = this._primary;
    if (old !== null && old.version === candidate.version) {
      return this.swapRecord("promote", old, candidate, false, 0, candidate.loadMs);
    }
    this._previous = old;
    this._primary = candidate; // single reference assignment: the swap itself
    if (this._shadow !== null && this._shadow.version === candidate.version) this._shadow = null;
    this.refreshRoleGauges();
    this.metrics.versionSwaps.labels({ kind: "promote" }).inc();
    const micros = (performance.now() - t0) * 1000;
    return this.swapRecord("promote", old, candidate, true, micros, candidate.loadMs);
  }

  rollback(): SwapRecord {
    if (this._previous === null) throw new NoPrimaryError("nothing to roll back to");
    const t0 = performance.now();
    const old = this._primary;
    const next = this._previous;
    this._previous = old;
    this._primary = next;
    this.refreshRoleGauges();
    this.metrics.versionSwaps.labels({ kind: "rollback" }).inc();
    const micros = (performance.now() - t0) * 1000;
    return this.swapRecord("rollback", old, next, true, micros, 0);
  }

  setShadow(version: string | null): LoadedModel | null {
    if (version === null) {
      this._shadow = null;
      this.refreshRoleGauges();
      return null;
    }
    const candidate = this.load(version);
    this._shadow = candidate;
    this.refreshRoleGauges();
    return candidate;
  }

  private swapRecord(
    kind: SwapRecord["kind"],
    old: LoadedModel | null,
    next: LoadedModel,
    changed: boolean,
    micros: number,
    loadMs: number,
  ): SwapRecord {
    const record: SwapRecord = {
      kind,
      from: old ? old.version : null,
      to: next.version,
      changed,
      at: this.now(),
      swap_micros: Math.round(micros * 10) / 10,
      load_ms: Math.round(loadMs * 1000) / 1000,
    };
    if (changed) this.swapHistory.push(record);
    return record;
  }

  private refreshRoleGauges(): void {
    for (const version of this.availableVersions()) {
      this.metrics.modelVersionInfo
        .labels({ version, role: "primary" })
        .set(this._primary?.version === version ? 1 : 0);
      this.metrics.modelVersionInfo
        .labels({ version, role: "shadow" })
        .set(this._shadow?.version === version ? 1 : 0);
    }
  }

  describe() {
    return {
      primary: this._primary?.version ?? null,
      shadow: this._shadow?.version ?? null,
      previous: this._previous?.version ?? null,
      versions: this.availableVersions().map((version) => ({
        version,
        file: MANIFEST.versions[version].file,
        test_mae_minutes: MANIFEST.versions[version].metrics.test_mae_minutes,
        loaded: this.loaded.has(version),
        role: this.roleOf(version),
      })),
      swaps: [...this.swapHistory],
    };
  }

  private roleOf(version: string): "primary" | "shadow" | null {
    if (this._primary?.version === version) return "primary";
    if (this._shadow?.version === version) return "shadow";
    return null;
  }
}
