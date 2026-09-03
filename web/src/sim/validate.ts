/**
 * Input checks: port of the strict pydantic schema in
 * modelgate/serving/schemas.py plus the reason mapping used for the
 * `modelgate_input_rejections_total{reason}` label.
 *
 * Strict means no coercion: "5" is not an int, 1 is not a bool, NaN and
 * Infinity are not finite, unknown fields are rejected, missing fields are
 * rejected, and a zone the model was not trained on is rejected.
 */

import { MAX_DISTANCE_KM, TRIP_FIELDS, ZONE_IDS, type Trip } from "./features";

export type RejectionReason =
  | "out_of_range"
  | "not_finite"
  | "wrong_type"
  | "unknown_zone"
  | "unknown_field"
  | "missing_field"
  | "malformed_body";

export const REJECTION_REASONS: readonly RejectionReason[] = [
  "out_of_range",
  "not_finite",
  "wrong_type",
  "unknown_zone",
  "unknown_field",
  "missing_field",
  "malformed_body",
];

export interface Rejection {
  field: string;
  reason: RejectionReason;
  message: string;
}

export type ValidationResult =
  | { ok: true; trip: Trip }
  | { ok: false; rejections: Rejection[] };

const ZONE_SET = new Set(ZONE_IDS);

function isPlainObject(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function typeName(v: unknown): string {
  if (v === null) return "null";
  if (Array.isArray(v)) return "array";
  if (typeof v === "number" && Number.isNaN(v)) return "NaN";
  return typeof v;
}

function checkFloat(field: string, v: unknown, min: number, max: number, out: Rejection[]): void {
  if (typeof v !== "number") {
    out.push({ field, reason: "wrong_type", message: `Input should be a valid number, got ${typeName(v)}` });
    return;
  }
  if (!Number.isFinite(v)) {
    out.push({ field, reason: "not_finite", message: "Input should be a finite number" });
    return;
  }
  if (v < min) out.push({ field, reason: "out_of_range", message: `Input should be greater than or equal to ${min}` });
  else if (v > max) out.push({ field, reason: "out_of_range", message: `Input should be less than or equal to ${max}` });
}

function checkInt(field: string, v: unknown, min: number, max: number, out: Rejection[]): void {
  if (typeof v !== "number" || !Number.isInteger(v)) {
    out.push({ field, reason: "wrong_type", message: `Input should be a valid integer, got ${typeName(v)}` });
    return;
  }
  if (v < min) out.push({ field, reason: "out_of_range", message: `Input should be greater than or equal to ${min}` });
  else if (v > max) out.push({ field, reason: "out_of_range", message: `Input should be less than or equal to ${max}` });
}

/** Validate an already-parsed body. Order of checks mirrors the field order. */
export function validateBody(body: unknown): ValidationResult {
  if (!isPlainObject(body)) {
    return {
      ok: false,
      rejections: [{ field: "body", reason: "malformed_body", message: "Input should be a JSON object" }],
    };
  }
  const rejections: Rejection[] = [];
  const known = new Set<string>(TRIP_FIELDS);

  for (const key of Object.keys(body)) {
    if (!known.has(key)) {
      rejections.push({ field: key, reason: "unknown_field", message: "Extra inputs are not permitted" });
    }
  }
  for (const field of TRIP_FIELDS) {
    if (!(field in body)) rejections.push({ field, reason: "missing_field", message: "Field required" });
  }

  if ("distance_km" in body) checkFloat("distance_km", body.distance_km, 0, MAX_DISTANCE_KM, rejections);
  if ("hour_of_day" in body) checkInt("hour_of_day", body.hour_of_day, 0, 23, rejections);
  if ("day_of_week" in body) checkInt("day_of_week", body.day_of_week, 0, 6, rejections);
  if ("pickup_zone_id" in body) {
    const z = body.pickup_zone_id;
    if (typeof z !== "number" || !Number.isInteger(z)) {
      rejections.push({
        field: "pickup_zone_id",
        reason: "wrong_type",
        message: `Input should be a valid integer, got ${typeName(z)}`,
      });
    } else if (!ZONE_SET.has(z)) {
      rejections.push({
        field: "pickup_zone_id",
        reason: "unknown_zone",
        message: `pickup_zone_id ${z} is not a known zone; expected one of ${ZONE_IDS.join(", ")}`,
      });
    }
  }
  if ("traffic_index" in body) checkFloat("traffic_index", body.traffic_index, 0, 1, rejections);
  if ("is_raining" in body && typeof body.is_raining !== "boolean") {
    rejections.push({
      field: "is_raining",
      reason: "wrong_type",
      message: `Input should be a valid boolean, got ${typeName(body.is_raining)}`,
    });
  }

  if (rejections.length) return { ok: false, rejections };
  return {
    ok: true,
    trip: {
      distance_km: body.distance_km as number,
      hour_of_day: body.hour_of_day as number,
      day_of_week: body.day_of_week as number,
      pickup_zone_id: body.pickup_zone_id as number,
      traffic_index: body.traffic_index as number,
      is_raining: body.is_raining as boolean,
    },
  };
}

const NAN_TOKEN = "__mg_nan__";
const POS_INF_TOKEN = "__mg_pinf__";
const NEG_INF_TOKEN = "__mg_ninf__";

function reviveSpecial(_key: string, value: unknown): unknown {
  if (value === NAN_TOKEN) return Number.NaN;
  if (value === POS_INF_TOKEN) return Number.POSITIVE_INFINITY;
  if (value === NEG_INF_TOKEN) return Number.NEGATIVE_INFINITY;
  return value;
}

/**
 * Parse raw request text the way the Python side sees it: Python's JSON
 * parser accepts bare NaN / Infinity tokens, which then fail the finite check.
 * Anything else that is not valid JSON is a malformed_body rejection.
 */
export function parseBody(text: string): ValidationResult {
  const prepared = text
    .replace(/(?<![\w"])NaN(?![\w"])/g, `"${NAN_TOKEN}"`)
    .replace(/(?<![\w"])-Infinity(?![\w"])/g, `"${NEG_INF_TOKEN}"`)
    .replace(/(?<![\w"-])Infinity(?![\w"])/g, `"${POS_INF_TOKEN}"`);
  let parsed: unknown;
  try {
    parsed = JSON.parse(prepared, reviveSpecial);
  } catch (err) {
    const msg = err instanceof Error ? err.message : "invalid JSON";
    return { ok: false, rejections: [{ field: "body", reason: "malformed_body", message: msg }] };
  }
  return validateBody(parsed);
}
