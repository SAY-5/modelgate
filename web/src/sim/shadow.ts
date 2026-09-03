/** Shadow-run bookkeeping: port of modelgate/serving/shadow.py. */

export interface ShadowRecord {
  request_id: string;
  primary_version: string;
  shadow_version: string;
  primary_eta: number;
  shadow_eta: number;
  at: number; // sim seconds
}

export function absDelta(r: ShadowRecord): number {
  return Math.abs(r.shadow_eta - r.primary_eta);
}

export function relDelta(r: ShadowRecord): number {
  return absDelta(r) / Math.max(r.primary_eta, 1e-6);
}

/** Nearest-rank percentile, same as the Python helper. */
export function percentile(values: number[], pct: number): number {
  if (!values.length) return 0;
  const ordered = [...values].sort((a, b) => a - b);
  const k = Math.max(0, Math.min(ordered.length - 1, Math.ceil((pct / 100) * ordered.length) - 1));
  return ordered[k];
}

export interface ShadowReport {
  count: number;
  errors: number;
  window: number;
  threshold_minutes: number;
  abs_delta_minutes: { mean: number; p50: number; p95: number; max: number };
  rel_delta: { mean: number; p95: number };
  share_beyond_threshold: number;
  shadow_bias_minutes: number;
  last: { request_id: string; primary: number; shadow: number; at: number } | null;
}

const round4 = (v: number) => Math.round(v * 1e4) / 1e4;

export class ShadowTracker {
  private records: ShadowRecord[] = [];
  private head = 0;
  private total = 0;
  private errors = 0;

  constructor(
    readonly thresholdMinutes: number,
    readonly maxRecords: number,
  ) {}

  record(rec: ShadowRecord): void {
    if (this.records.length < this.maxRecords) {
      this.records.push(rec);
    } else {
      this.records[this.head] = rec;
      this.head = (this.head + 1) % this.maxRecords;
    }
    this.total += 1;
  }

  recordError(): void {
    this.errors += 1;
  }

  reset(): void {
    this.records = [];
    this.head = 0;
    this.total = 0;
    this.errors = 0;
  }

  /** Records in insertion order (oldest first). */
  window(): ShadowRecord[] {
    if (this.records.length < this.maxRecords) return [...this.records];
    return [...this.records.slice(this.head), ...this.records.slice(0, this.head)];
  }

  report(): ShadowReport {
    const records = this.window();
    const n = records.length;
    const abs = records.map(absDelta);
    const rel = records.map(relDelta);
    const beyond = abs.filter((d) => d > this.thresholdMinutes).length;
    const sum = (xs: number[]) => xs.reduce((a, b) => a + b, 0);
    const last = records[n - 1];
    return {
      count: this.total,
      errors: this.errors,
      window: n,
      threshold_minutes: this.thresholdMinutes,
      abs_delta_minutes: {
        mean: n ? round4(sum(abs) / n) : 0,
        p50: round4(percentile(abs, 50)),
        p95: round4(percentile(abs, 95)),
        max: n ? round4(Math.max(...abs)) : 0,
      },
      rel_delta: { mean: n ? round4(sum(rel) / n) : 0, p95: round4(percentile(rel, 95)) },
      share_beyond_threshold: n ? round4(beyond / n) : 0,
      shadow_bias_minutes: n ? round4(sum(records.map((r) => r.shadow_eta - r.primary_eta)) / n) : 0,
      last: last
        ? { request_id: last.request_id, primary: last.primary_eta, shadow: last.shadow_eta, at: last.at }
        : null,
    };
  }
}
