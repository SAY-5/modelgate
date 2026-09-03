/**
 * Prometheus-style metrics: counters, gauges, and histograms with labels,
 * plus a text exposition renderer. Port of modelgate/serving/metrics.py.
 */

export type Labels = Record<string, string>;

function labelKey(labels: Labels): string {
  return Object.keys(labels)
    .sort()
    .map((k) => `${k}=${JSON.stringify(labels[k])}`)
    .join(",");
}

function renderLabels(labels: Labels, extra?: Labels): string {
  const all = { ...labels, ...(extra ?? {}) };
  const keys = Object.keys(all);
  if (!keys.length) return "";
  return `{${keys.map((k) => `${k}="${all[k]}"`).join(",")}}`;
}

abstract class Collector {
  constructor(
    readonly name: string,
    readonly help: string,
    readonly labelNames: readonly string[],
  ) {}
  abstract readonly type: "counter" | "gauge" | "histogram";
  abstract expose(): string[];
}

export class Counter extends Collector {
  readonly type = "counter" as const;
  private readonly values = new Map<string, { labels: Labels; value: number }>();

  labels(labels: Labels = {}): { inc: (by?: number) => void } {
    const key = labelKey(labels);
    let cell = this.values.get(key);
    if (!cell) {
      cell = { labels, value: 0 };
      this.values.set(key, cell);
    }
    const c = cell;
    return { inc: (by = 1) => void (c.value += by) };
  }

  inc(by = 1): void {
    this.labels({}).inc(by);
  }

  get(labels: Labels = {}): number {
    return this.values.get(labelKey(labels))?.value ?? 0;
  }

  series(): { labels: Labels; value: number }[] {
    return [...this.values.values()];
  }

  expose(): string[] {
    const lines = [`# HELP ${this.name} ${this.help}`, `# TYPE ${this.name} counter`];
    for (const s of this.values.values()) lines.push(`${this.name}${renderLabels(s.labels)} ${s.value}`);
    return lines;
  }
}

export class Gauge extends Collector {
  readonly type = "gauge" as const;
  private readonly values = new Map<string, { labels: Labels; value: number }>();

  labels(labels: Labels = {}): { set: (v: number) => void } {
    const key = labelKey(labels);
    let cell = this.values.get(key);
    if (!cell) {
      cell = { labels, value: 0 };
      this.values.set(key, cell);
    }
    const c = cell;
    return { set: (v: number) => void (c.value = v) };
  }

  set(v: number): void {
    this.labels({}).set(v);
  }

  get(labels: Labels = {}): number {
    return this.values.get(labelKey(labels))?.value ?? 0;
  }

  series(): { labels: Labels; value: number }[] {
    return [...this.values.values()];
  }

  expose(): string[] {
    const lines = [`# HELP ${this.name} ${this.help}`, `# TYPE ${this.name} gauge`];
    for (const s of this.values.values()) lines.push(`${this.name}${renderLabels(s.labels)} ${s.value}`);
    return lines;
  }
}

export interface HistogramChild {
  labels: Labels;
  buckets: number[]; // non-cumulative counts per bucket, last is +Inf
  sum: number;
  count: number;
}

export class Histogram extends Collector {
  readonly type = "histogram" as const;
  private readonly children = new Map<string, HistogramChild>();

  constructor(name: string, help: string, labelNames: readonly string[], readonly bounds: readonly number[]) {
    super(name, help, labelNames);
  }

  labels(labels: Labels = {}): { observe: (v: number) => void } {
    return { observe: (v: number) => this.observe(v, labels) };
  }

  observe(v: number, labels: Labels = {}): void {
    const child = this.child(labels);
    let i = 0;
    while (i < this.bounds.length && v > this.bounds[i]) i++;
    child.buckets[i] += 1;
    child.sum += v;
    child.count += 1;
  }

  child(labels: Labels = {}): HistogramChild {
    const key = labelKey(labels);
    let c = this.children.get(key);
    if (!c) {
      c = { labels, buckets: new Array(this.bounds.length + 1).fill(0), sum: 0, count: 0 };
      this.children.set(key, c);
    }
    return c;
  }

  series(): HistogramChild[] {
    return [...this.children.values()];
  }

  count(labels: Labels = {}): number {
    return this.children.get(labelKey(labels))?.count ?? 0;
  }

  /** Cumulative bucket counts, the way Prometheus exposes them. */
  cumulative(child: HistogramChild): number[] {
    const out: number[] = [];
    let acc = 0;
    for (const b of child.buckets) {
      acc += b;
      out.push(acc);
    }
    return out;
  }

  /**
   * Quantile estimate with linear interpolation inside the bucket, the same
   * estimator PromQL's histogram_quantile uses.
   */
  quantile(q: number, child: HistogramChild): number {
    if (child.count === 0) return 0;
    const cum = this.cumulative(child);
    const rank = q * child.count;
    let i = 0;
    while (i < cum.length - 1 && cum[i] < rank) i++;
    if (i === cum.length - 1) return this.bounds[this.bounds.length - 1];
    const lower = i === 0 ? 0 : this.bounds[i - 1];
    const upper = this.bounds[i];
    const prev = i === 0 ? 0 : cum[i - 1];
    const inBucket = cum[i] - prev;
    if (inBucket === 0) return upper;
    return lower + ((upper - lower) * (rank - prev)) / inBucket;
  }

  expose(): string[] {
    const lines = [`# HELP ${this.name} ${this.help}`, `# TYPE ${this.name} histogram`];
    for (const c of this.children.values()) {
      const cum = this.cumulative(c);
      this.bounds.forEach((le, i) => {
        lines.push(`${this.name}_bucket${renderLabels(c.labels, { le: String(le) })} ${cum[i]}`);
      });
      lines.push(`${this.name}_bucket${renderLabels(c.labels, { le: "+Inf" })} ${c.count}`);
      lines.push(`${this.name}_sum${renderLabels(c.labels)} ${Number(c.sum.toPrecision(8))}`);
      lines.push(`${this.name}_count${renderLabels(c.labels)} ${c.count}`);
    }
    return lines;
  }
}

export const LATENCY_BUCKETS = [0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5];
export const ETA_BUCKETS = [1, 2, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 300];
export const DIVERGENCE_BUCKETS = [0.1, 0.25, 0.5, 1, 2, 3, 5, 10, 20, 60];

/** One metric set per service instance (the Python module uses one per process). */
export class MetricSet {
  readonly requests = new Counter(
    "modelgate_requests_total",
    "Prediction requests by serving model version and outcome.",
    ["version", "outcome"],
  );
  readonly requestLatency = new Histogram(
    "modelgate_request_latency_seconds",
    "End-to-end /predict latency including validation and shadow run.",
    ["version"],
    LATENCY_BUCKETS,
  );
  readonly predictionsEta = new Histogram(
    "modelgate_predictions_eta_minutes",
    "Distribution of ETA values returned to clients.",
    ["version"],
    ETA_BUCKETS,
  );
  readonly inputRejections = new Counter(
    "modelgate_input_rejections_total",
    "Requests rejected by input validation, by reason.",
    ["reason"],
  );
  readonly shadowDivergence = new Histogram(
    "modelgate_shadow_divergence_minutes",
    "Absolute difference between shadow and primary ETA in minutes.",
    ["primary", "shadow"],
    DIVERGENCE_BUCKETS,
  );
  readonly shadowRequests = new Counter(
    "modelgate_shadow_requests_total",
    "Shadow inferences executed, by outcome.",
    ["shadow", "outcome"],
  );
  readonly modelVersionInfo = new Gauge(
    "modelgate_model_version_info",
    "1 for the version currently serving in the given role (primary or shadow).",
    ["version", "role"],
  );
  readonly versionSwaps = new Counter(
    "modelgate_version_swaps_total",
    "Primary version swaps (promotions and rollbacks).",
    ["kind"],
  );
  readonly droppedRequests = new Counter(
    "modelgate_dropped_requests_total",
    "Prediction requests that failed for a reason other than invalid input. Must stay 0.",
    [],
  );
  readonly modelsLoaded = new Gauge(
    "modelgate_models_loaded",
    "Number of model versions resident in memory.",
    [],
  );

  all(): Collector[] {
    return [
      this.requests,
      this.requestLatency,
      this.predictionsEta,
      this.inputRejections,
      this.shadowDivergence,
      this.shadowRequests,
      this.modelVersionInfo,
      this.versionSwaps,
      this.droppedRequests,
      this.modelsLoaded,
    ];
  }

  /** Prometheus text exposition, like GET /metrics. */
  expose(): string {
    return this.all()
      .flatMap((c) => c.expose())
      .join("\n");
  }
}
