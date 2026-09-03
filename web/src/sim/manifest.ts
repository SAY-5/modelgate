/** Mirror of artifacts/manifest.json: the numbers the real training run produced. */

import { FEATURE_DIM, ZONE_IDS, featureNames } from "./features";

export interface VersionEntry {
  file: string;
  arch: { hidden: number; depth: number };
  training: { epochs: number; lr: number };
  metrics: { test_mae_minutes: number };
  train_seconds: number;
}

export const MANIFEST = {
  modelgate_version: "0.1.0",
  seed: 7,
  dataset: { size: 12000, test_fraction: 0.2 },
  feature_schema: {
    dim: FEATURE_DIM,
    names: featureNames(),
    zone_ids: [...ZONE_IDS],
    inputs: {
      distance_km: { type: "float", min: 0, max: 500 },
      hour_of_day: { type: "int", min: 0, max: 23 },
      day_of_week: { type: "int", min: 0, max: 6 },
      pickup_zone_id: { type: "int", enum: [...ZONE_IDS] },
      traffic_index: { type: "float", min: 0, max: 1 },
      is_raining: { type: "bool" },
    },
  },
  baselines: { mean_predictor_mae: 8.6508, distance_linear_mae: 3.5654 },
  versions: {
    v1: {
      file: "eta_v1.pt",
      arch: { hidden: 64, depth: 2 },
      training: { epochs: 10, lr: 0.003 },
      metrics: { test_mae_minutes: 3.1051 },
      train_seconds: 1.04,
    },
    v2: {
      file: "eta_v2.pt",
      arch: { hidden: 96, depth: 3 },
      training: { epochs: 25, lr: 0.002 },
      metrics: { test_mae_minutes: 1.888 },
      train_seconds: 0.65,
    },
  } as Record<string, VersionEntry>,
} as const;

export type VersionId = keyof typeof MANIFEST.versions & string;

/** Headline numbers from the committed `make demo` run in the README. */
export const REAL_DEMO = {
  target_rps: 200,
  duration_s: 20,
  total_requests: 4000,
  successes: 4000,
  dropped: 0,
  server_dropped_metric: 0,
  latency_ms: { p50: 1.78, p95: 4.18, p99: 11.8, max: 63.91 },
  shadow_enabled_at_s: 5.001,
  swap_requested_at_s: 10.0,
  swap_completed_at_s: 10.006,
  versions_before: { v1: 2001, v2: 1 },
  versions_after: { v2: 1998 },
  shadow_report: { n: 1000, mean_abs: 2.2859, p95_abs: 6.3505, beyond_2min: 0.401 },
} as const;
