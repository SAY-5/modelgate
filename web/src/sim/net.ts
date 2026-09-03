/**
 * ETA regressor: the same MLP as modelgate/model/net.py, run on the real
 * weights exported from artifacts/eta_v1.pt and eta_v2.pt.
 *
 * forward(x) = softplus(Linear(ReLU(Linear(... x))))  with a single output.
 */

import { FEATURE_DIM } from "./features";
import rawWeights from "./weights.json";

export interface Layer {
  w: number[][];
  b: number[];
}

export interface NetSpec {
  hidden: number;
  depth: number;
  layers: Layer[];
}

export type WeightBundle = Record<string, NetSpec>;

export const WEIGHTS = rawWeights as WeightBundle;

function softplus(x: number): number {
  // Matches torch.nn.functional.softplus with the default threshold of 20.
  return x > 20 ? x : Math.log1p(Math.exp(x));
}

export class EtaNet {
  readonly hidden: number;
  readonly depth: number;
  readonly paramCount: number;
  private readonly layers: { w: Float32Array; b: Float32Array; out: number; inp: number }[];

  constructor(spec: NetSpec) {
    this.hidden = spec.hidden;
    this.depth = spec.depth;
    this.layers = spec.layers.map((l) => {
      const out = l.w.length;
      const inp = l.w[0].length;
      const w = new Float32Array(out * inp);
      for (let i = 0; i < out; i++) for (let j = 0; j < inp; j++) w[i * inp + j] = l.w[i][j];
      return { w, b: Float32Array.from(l.b), out, inp };
    });
    if (this.layers[0].inp !== FEATURE_DIM) {
      throw new Error(`feature dim mismatch: net expects ${this.layers[0].inp}, got ${FEATURE_DIM}`);
    }
    this.paramCount = this.layers.reduce((n, l) => n + l.w.length + l.b.length, 0);
  }

  /** Single-row forward pass. Returns ETA in minutes (always positive). */
  forward(x: Float32Array): number {
    let cur = x;
    const last = this.layers.length - 1;
    for (let li = 0; li <= last; li++) {
      const { w, b, out, inp } = this.layers[li];
      const next = new Float32Array(out);
      for (let i = 0; i < out; i++) {
        let acc = b[i];
        const base = i * inp;
        for (let j = 0; j < inp; j++) acc += w[base + j] * cur[j];
        next[i] = li === last ? acc : acc > 0 ? acc : 0;
      }
      cur = next;
    }
    return softplus(cur[0]);
  }
}
