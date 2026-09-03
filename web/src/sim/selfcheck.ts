/**
 * Console self-check for the browser port. Run with `npm run selfcheck`
 * (Node via tsx) or import `runSelfCheck` in the browser console.
 *
 * 1. v1 and v2 reproduce the torch reference predictions and differ.
 * 2. The input rules reject each malformed case with the right reason.
 * 3. A 200 rps burst across a promote yields 0 dropped requests.
 */

import { encode, type Trip } from "./features";
import { LoadGen } from "./loadgen";
import { Service } from "./service";
import { validateBody } from "./validate";

export interface CheckLine {
  name: string;
  pass: boolean;
  detail: string;
}

// Values printed by torch on the committed artifacts when the weights were exported.
const REFERENCE: { trip: Trip; v1: number; v2: number }[] = [
  {
    trip: { distance_km: 5, hour_of_day: 8, day_of_week: 1, pickup_zone_id: 3, traffic_index: 0.7, is_raining: false },
    v1: 17.8823,
    v2: 17.2121,
  },
  {
    trip: { distance_km: 20, hour_of_day: 23, day_of_week: 6, pickup_zone_id: 11, traffic_index: 0.1, is_raining: true },
    v1: 32.0781,
    v2: 38.5474,
  },
  {
    trip: { distance_km: 0, hour_of_day: 0, day_of_week: 0, pickup_zone_id: 1, traffic_index: 0, is_raining: false },
    v1: 4.8207,
    v2: 3.0175,
  },
];

export function runSelfCheck(): CheckLine[] {
  const lines: CheckLine[] = [];
  const svc = new Service({ seed: 7 });
  svc.boot("v1");
  const v1 = svc.registry.load("v1");
  const v2 = svc.registry.load("v2");

  for (const ref of REFERENCE) {
    const f = encode(ref.trip);
    const p1 = v1.predict(f);
    const p2 = v2.predict(f);
    const close = Math.abs(p1 - ref.v1) < 2e-3 && Math.abs(p2 - ref.v2) < 2e-3;
    lines.push({
      name: `torch parity ${ref.trip.distance_km}km z${ref.trip.pickup_zone_id}`,
      pass: close && Math.abs(p1 - p2) > 0.05,
      detail: `v1 ${p1.toFixed(4)} (ref ${ref.v1}) v2 ${p2.toFixed(4)} (ref ${ref.v2})`,
    });
  }

  const good: Trip = { distance_km: 7.5, hour_of_day: 17, day_of_week: 4, pickup_zone_id: 6, traffic_index: 0.8, is_raining: true };
  const cases: [string, unknown, string][] = [
    ["out_of_range distance", { ...good, distance_km: 900 }, "out_of_range"],
    ["not_finite distance", { ...good, distance_km: Number.NaN }, "not_finite"],
    ["wrong_type hour", { ...good, hour_of_day: "17" }, "wrong_type"],
    ["wrong_type is_raining=1", { ...good, is_raining: 1 }, "wrong_type"],
    ["unknown_zone 13", { ...good, pickup_zone_id: 13 }, "unknown_zone"],
    ["unknown_field", { ...good, driver_tip: 3 }, "unknown_field"],
    ["missing_field", (({ traffic_index: _t, ...rest }) => rest)(good), "missing_field"],
    ["malformed_body", "not an object", "malformed_body"],
  ];
  for (const [name, body, reason] of cases) {
    const r = validateBody(body);
    const got = r.ok ? "accepted" : r.rejections.map((x) => x.reason).join(",");
    lines.push({ name: `reject ${name}`, pass: !r.ok && got.includes(reason), detail: got });
  }
  const accepted = validateBody(good);
  lines.push({ name: "accept valid trip", pass: accepted.ok, detail: accepted.ok ? "ok" : "rejected" });

  // Burst across a swap: 200 rps for 4 s in 20 ms ticks, promote at t+2 s.
  const gen = new LoadGen(svc, { rps: 200, seed: 1 });
  gen.running = true;
  svc.setShadow("v2");
  let promoted = false;
  let swapMicros = 0;
  for (let step = 0; step < 200; step++) {
    if (!promoted && gen.elapsed >= 2) {
      swapMicros = svc.promoteNow("v2").swap_micros;
      promoted = true;
    }
    gen.tick(0.02);
  }
  gen.drain();
  const st = gen.stats();
  const before = svc.metrics.requests.get({ version: "v1", outcome: "ok" });
  const after = svc.metrics.requests.get({ version: "v2", outcome: "ok" });
  lines.push({
    name: "zero drop across promote",
    pass: st.dropped === 0 && st.ok === st.sent && svc.metrics.droppedRequests.get() === 0 && promoted,
    detail: `sent ${st.sent} ok ${st.ok} dropped ${st.dropped} server_dropped ${svc.metrics.droppedRequests.get()} v1 ${before} v2 ${after} swap ${swapMicros}us`,
  });
  const rep = svc.shadowReport();
  lines.push({
    name: "shadow report populated",
    pass: rep.count > 0 && rep.abs_delta_minutes.mean > 0,
    detail: `n=${rep.count} mean|d|=${rep.abs_delta_minutes.mean} p95|d|=${rep.abs_delta_minutes.p95} beyond2=${rep.share_beyond_threshold}`,
  });
  const rb = svc.rollback();
  lines.push({
    name: "rollback restores v1",
    pass: rb.to === "v1" && svc.registry.primary?.version === "v1",
    detail: `${rb.from} -> ${rb.to} in ${rb.swap_micros}us`,
  });
  return lines;
}

export function formatSelfCheck(lines: CheckLine[]): string {
  const w = Math.max(...lines.map((l) => l.name.length));
  return lines.map((l) => `${l.pass ? "PASS" : "FAIL"}  ${l.name.padEnd(w)}  ${l.detail}`).join("\n");
}

// Node entry point (tsx). Guarded so the browser bundle ignores it.
declare const process: { argv?: string[]; exitCode?: number } | undefined;
if (typeof process !== "undefined" && process?.argv?.[1]?.endsWith("selfcheck.ts")) {
  const lines = runSelfCheck();
  console.log(formatSelfCheck(lines));
  if (lines.some((l) => !l.pass)) process.exitCode = 1;
}
