import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { useMemo, useState } from "react";
import { TRIP_FIELDS, ZONE_IDS } from "../sim/features";
import type { PredictOutcome } from "../sim/service";
import { REJECTION_REASONS } from "../sim/validate";
import { useService } from "../state/ServiceProvider";
import { BucketHistogram, BucketLabels } from "./charts/Bars";

type FieldKey = (typeof TRIP_FIELDS)[number];

/** Each field is a raw JSON literal so the strict rules can be provoked directly. */
type Raw = Record<FieldKey, string>;

const DEFAULTS: Raw = {
  distance_km: "7.5",
  hour_of_day: "17",
  day_of_week: "4",
  pickup_zone_id: "6",
  traffic_index: "0.8",
  is_raining: "true",
};

const RULES: Record<FieldKey, string> = {
  distance_km: "float 0..500, finite",
  hour_of_day: "int 0..23",
  day_of_week: "int 0..6",
  pickup_zone_id: `int in {${ZONE_IDS[0]}..${ZONE_IDS[ZONE_IDS.length - 1]}}`,
  traffic_index: "float 0..1, finite",
  is_raining: "bool, strict",
};

interface Preset {
  label: string;
  hint: string;
  raw?: Partial<Raw>;
  extra?: string;
  omit?: FieldKey;
}

const PRESETS: Preset[] = [
  { label: "Valid trip", hint: "200", raw: DEFAULTS },
  { label: "NaN distance", hint: "not_finite", raw: { distance_km: "NaN" } },
  { label: "Hour as string", hint: "wrong_type", raw: { hour_of_day: '"17"' } },
  { label: "Rain = 1", hint: "wrong_type", raw: { is_raining: "1" } },
  { label: "Zone 13", hint: "unknown_zone", raw: { pickup_zone_id: "13" } },
  { label: "620 km", hint: "out_of_range", raw: { distance_km: "620" } },
  { label: "Extra field", hint: "unknown_field", extra: '"driver_tip": 3' },
  { label: "Drop traffic", hint: "missing_field", omit: "traffic_index" },
  { label: "Broken JSON", hint: "malformed_body", raw: { traffic_index: "0.8," } },
];

interface Sent {
  id: number;
  body: string;
  outcome: PredictOutcome;
}

export function InputChecks() {
  const { service, refresh } = useService();
  const reduce = useReducedMotion();
  const [raw, setRaw] = useState<Raw>(DEFAULTS);
  const [extra, setExtra] = useState("");
  const [omit, setOmit] = useState<FieldKey | null>(null);
  const [history, setHistory] = useState<Sent[]>([]);
  const [seq, setSeq] = useState(1);

  const body = useMemo(() => {
    const lines = TRIP_FIELDS.filter((f) => f !== omit).map((f) => `  "${f}": ${raw[f].trim() || "null"}`);
    if (extra.trim()) lines.push(`  ${extra.trim()}`);
    return `{\n${lines.join(",\n")}\n}`;
  }, [raw, extra, omit]);

  const latest = history[0];
  const fieldErrors = useMemo(() => {
    const m = new Map<string, string>();
    if (latest?.outcome.status === 422) for (const r of latest.outcome.rejections) m.set(r.field, r.reason);
    return m;
  }, [latest]);

  const send = () => {
    const outcome = service.predict(body);
    setHistory((h) => [{ id: seq, body, outcome }, ...h].slice(0, 8));
    setSeq((s) => s + 1);
    refresh();
  };

  const applyPreset = (p: Preset) => {
    setRaw(p.raw === DEFAULTS ? DEFAULTS : { ...DEFAULTS, ...(p.raw ?? {}) });
    setExtra(p.extra ?? "");
    setOmit(p.omit ?? null);
  };

  const rejections = service.metrics.inputRejections;
  const bars = REJECTION_REASONS.map((r) => ({ label: r, value: rejections.get({ reason: r }) }));
  const totalRejected = bars.reduce((a, b) => a + b.value, 0);

  return (
    <section className="section" id="checks" aria-labelledby="checks-title">
      <div className="container">
        <div className="section-head">
          <span className="section-index">01 / input checks</span>
          <h2 className="section-title" id="checks-title">
            Nothing reaches the tensor unchecked.
          </h2>
          <p className="section-lede">
            Every field is a raw JSON literal here, so you can send exactly what a misbehaving client
            would. Strict mode means no coercion: <code>"17"</code> is not an int, <code>1</code> is not a
            bool, <code>NaN</code> is not finite, and a zone the model never saw is refused before any
            feature is built. Each rejection is a 422 with a per-field reason and bumps a counter.
          </p>
        </div>

        <div className="checks-grid">
          <div className="glass panel builder">
            <div className="builder-head">
              <span className="eyebrow">POST /predict</span>
              <span className="pill">strict=True, extra=forbid</span>
            </div>
            <div className="builder-fields">
              {TRIP_FIELDS.map((f) => {
                const err = fieldErrors.get(f);
                const omitted = omit === f;
                return (
                  <div key={f} className={`field ${err ? "invalid" : ""} ${omitted ? "omitted" : ""}`}>
                    <label htmlFor={`f-${f}`}>{f}</label>
                    <input
                      id={`f-${f}`}
                      value={raw[f]}
                      disabled={omitted}
                      spellCheck={false}
                      autoComplete="off"
                      onChange={(e) => setRaw((r) => ({ ...r, [f]: e.target.value }))}
                      aria-describedby={`h-${f}`}
                      aria-invalid={err ? true : undefined}
                    />
                    <span id={`h-${f}`} className={err ? "err" : "hint"}>
                      {omitted ? "omitted from body" : err ? err : RULES[f]}
                    </span>
                  </div>
                );
              })}
              <div className={`field field-wide ${fieldErrors.has(extra.split(":")[0]?.replace(/"/g, "").trim()) ? "invalid" : ""}`}>
                <label htmlFor="f-extra">extra key (raw)</label>
                <input
                  id="f-extra"
                  value={extra}
                  placeholder='"driver_tip": 3'
                  spellCheck={false}
                  autoComplete="off"
                  onChange={(e) => setExtra(e.target.value)}
                />
                <span className="hint">any unknown key is rejected, not ignored</span>
              </div>
            </div>

            <div className="builder-presets" role="group" aria-label="Presets">
              {PRESETS.map((p) => (
                <button key={p.label} type="button" className="chip" onClick={() => applyPreset(p)}>
                  {p.label} <span className="chip-hint mono">{p.hint}</span>
                </button>
              ))}
              {omit && (
                <button type="button" className="chip" onClick={() => setOmit(null)}>
                  restore {omit}
                </button>
              )}
            </div>

            <pre className="body-preview mono" aria-label="Request body">
              {body}
            </pre>
            <div className="builder-actions">
              <button type="button" className="btn btn-primary" onClick={send}>
                Send request
              </button>
              <span className="mono t-dim">
                primary {service.registry.primary?.version ?? "none"}
              </span>
            </div>
          </div>

          <div className="checks-side">
            <div className="glass panel response" aria-live="polite">
              <div className="builder-head">
                <span className="eyebrow">response</span>
                {latest ? (
                  <span className={`pill ${latest.outcome.status === 200 ? "ok" : "bad"}`}>
                    <span className="dot" /> {latest.outcome.status}{" "}
                    {latest.outcome.status === 200 ? "OK" : latest.outcome.status === 422 ? "Unprocessable" : "Error"}
                  </span>
                ) : (
                  <span className="pill">waiting</span>
                )}
              </div>
              <AnimatePresence mode="wait" initial={false}>
                {latest ? (
                  <motion.div
                    key={latest.id}
                    initial={reduce ? false : { opacity: 0, y: 6 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={reduce ? undefined : { opacity: 0, y: -6 }}
                    transition={{ duration: 0.22 }}
                  >
                    {latest.outcome.status === 200 ? (
                      <div className="eta-card">
                        <div className="stat">
                          <span className="k">eta_minutes</span>
                          <span className="v orange mono">{latest.outcome.eta_minutes.toFixed(2)}</span>
                          <span className="sub">
                            model_version {latest.outcome.model_version} / request_id {latest.outcome.request_id}
                          </span>
                        </div>
                        {latest.outcome.shadow && (
                          <div className="stat">
                            <span className="k">shadow {latest.outcome.shadow.version} (not returned)</span>
                            <span className="v ice mono">{latest.outcome.shadow.eta_minutes.toFixed(2)}</span>
                            <span className="sub">abs delta {latest.outcome.shadow.abs_delta.toFixed(2)} min</span>
                          </div>
                        )}
                        <pre className="mono response-json">
                          {JSON.stringify(
                            {
                              eta_minutes: latest.outcome.eta_minutes,
                              model_version: latest.outcome.model_version,
                              request_id: latest.outcome.request_id,
                            },
                            null,
                            2,
                          )}
                        </pre>
                      </div>
                    ) : latest.outcome.status === 422 ? (
                      <div>
                        <ul className="rejections">
                          {latest.outcome.rejections.map((r, i) => (
                            <li key={i}>
                              <span className="pill bad mono">{r.reason}</span>
                              <span className="mono rej-field">{r.field}</span>
                              <span className="rej-msg">{r.message}</span>
                            </li>
                          ))}
                        </ul>
                        <pre className="mono response-json">
                          {JSON.stringify(
                            { error: latest.outcome.error, rejections: latest.outcome.rejections },
                            null,
                            2,
                          )}
                        </pre>
                      </div>
                    ) : (
                      <pre className="mono response-json">{JSON.stringify(latest.outcome, null, 2)}</pre>
                    )}
                  </motion.div>
                ) : (
                  <p className="t-dim">Send a request to see the response and how it is counted.</p>
                )}
              </AnimatePresence>
            </div>

            <div className="glass panel">
              <div className="builder-head">
                <span className="eyebrow">modelgate_input_rejections_total</span>
                <span className="pill mono">{totalRejected} total</span>
              </div>
              <BucketHistogram
                bars={bars.map((b) => ({ ...b, highlight: fieldErrors.size > 0 && [...fieldErrors.values()].includes(b.label as never) }))}
                height={120}
                color="var(--bad)"
                highlightColor="var(--orange)"
                ariaLabel="Rejections by reason"
              />
              <BucketLabels labels={bars.map((b) => b.label.replace("_", " "))} />
              <p className="hint-line">
                Rejections count as <code>outcome="rejected"</code> in the request counter. They are never
                drops. The stream in section 03 can inject malformed inputs to show the difference.
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
