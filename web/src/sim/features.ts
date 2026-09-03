/**
 * Feature schema shared by training and serving.
 * Port of modelgate/model/features.py: the encoding must match the tensor
 * the committed artifacts were trained on, column for column.
 */

export const ZONE_IDS: readonly number[] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
export const NUM_ZONES = ZONE_IDS.length;

export const BASE_FEATURES = [
  "distance_km",
  "hour_sin",
  "hour_cos",
  "dow_sin",
  "dow_cos",
  "traffic_index",
  "is_raining",
] as const;

export const FEATURE_DIM = BASE_FEATURES.length + NUM_ZONES;
export const MAX_DISTANCE_KM = 500;
export const DISTANCE_SCALE_KM = 50;

export interface Trip {
  distance_km: number;
  hour_of_day: number;
  day_of_week: number;
  pickup_zone_id: number;
  traffic_index: number;
  is_raining: boolean;
}

export const TRIP_FIELDS: readonly (keyof Trip)[] = [
  "distance_km",
  "hour_of_day",
  "day_of_week",
  "pickup_zone_id",
  "traffic_index",
  "is_raining",
];

export function featureNames(): string[] {
  return [...BASE_FEATURES, ...ZONE_IDS.map((z) => `zone_${z}`)];
}

/** Encode one trip into the model's flat feature vector. */
export function encode(t: Trip): Float32Array {
  const hourAngle = (2 * Math.PI * t.hour_of_day) / 24;
  const dowAngle = (2 * Math.PI * t.day_of_week) / 7;
  const row = new Float32Array(FEATURE_DIM);
  row[0] = t.distance_km / DISTANCE_SCALE_KM;
  row[1] = Math.sin(hourAngle);
  row[2] = Math.cos(hourAngle);
  row[3] = Math.sin(dowAngle);
  row[4] = Math.cos(dowAngle);
  row[5] = t.traffic_index;
  row[6] = t.is_raining ? 1 : 0;
  row[BASE_FEATURES.length + ZONE_IDS.indexOf(t.pickup_zone_id)] = 1;
  return row;
}
