import { motion, useReducedMotion } from "framer-motion";
import { DIVERGENCE_BUCKETS } from "../sim/metrics";
import { MANIFEST } from "../sim/manifest";
import { useService } from "../state/ServiceProvider";
import { BucketHistogram, BucketLabels } from "./charts/Bars";

const fmt = (v: number, d = 2) => v.toFixed(d);

export function ShadowRuns() {
  const { service, gen, refresh } = useService();
  const reduce = useReducedMotion();
  const primary = service.registry.primary?.version ?? "v1";
  const candidate = service.registry.availableVersions().find((v) => v !== primary) ?? "v2";
  const shadow = service.registry.shadow?.version ?? null;
  const on = shadow !== null;
  const report = service.shadowReport();
  const child = service.metrics.shadowDivergence.child({ primary, shadow: shadow ?? candidate });
  const bounds = DIVERGENCE_BUCKETS;
  const bars = child.buckets.map((v, i) => ({
    label: i < bounds.length ? String(bounds[i]) : "inf",
    value: v,
    highlight: i < bounds.length ? bounds[i] > report.threshold_minutes : true,
  }));
  const last = report.last;
  const mae = (v: string) => MANIFEST.versions[v]?.metrics.test_mae_minutes;

  const toggle = () => {
    service.setShadow(on ? null : candidate);
    if (!on && !gen.running) {
      gen.running = true;
    }
    refresh();
  };

  return (
    <section className="section" id="shadow" aria-labelledby="shadow-title">
      <div className="container">
        <div className="section-head">
          <span className="section-index">02 / shadow runs</span>
          <h2 className="section-title" id="shadow-title">
            Let the candidate watch. <span className="t-ice">Never let it answer.</span>
          </h2>
          <p className="section-lede">
            With a shadow version set, every request runs the primary first, then the shadow on the same
            feature vector inside a try block. The response is built from the primary before the shadow
            runs. The shadow result only feeds a divergence histogram and a rolling report.
          </p>
        </div>

        <div className="shadow-grid">
          <div className="glass panel ice-edge shadow-control">
            <div className="role-row">
              <div className="role primary">
                <span className="eyebrow">primary, answers clients</span>
                <span className="role-name">{primary}</span>
                <span className="mono t-dim">held-out MAE {mae(primary)} min</span>
              </div>
              <div className={`role shadow ${on ? "on" : ""}`}>
                <span className="eyebrow">shadow, observed only</span>
                <span className="role-name">{on ? shadow : candidate}</span>
                <span className="mono t-dim">held-out MAE {mae(on ? shadow! : candidate)} min</span>
              </div>
            </div>
            <div className="toggle-row">
              <button
                type="button"
                role="switch"
                aria-checked={on}
                className="switch"
                id="shadow-switch"
                onClick={toggle}
              />
              <label htmlFor="shadow-switch" className="toggle-label">
                Shadow {candidate} on live traffic
                <span className="hint">
                  {gen.running
                    ? `${gen.rps} rps stream running`
                    : "starts the 200 rps stream from section 03"}
                </span>
              </label>
            </div>
            {last ? (
              <div className="last-record" aria-live="polite">
                <span className="eyebrow">last record {last.request_id}</span>
                <div className="last-pair">
                  <div className="stat">
                    <span className="k">{primary} served to client</span>
                    <span className="v orange mono">{fmt(last.primary)}</span>
                  </div>
                  <div className="stat">
                    <span className="k">{shadow} observed</span>
                    <span className="v ice mono">{fmt(last.shadow)}</span>
                  </div>
                  <div className="stat">
                    <span className="k">abs delta</span>
                    <span className={`v mono ${Math.abs(last.shadow - last.primary) > report.threshold_minutes ? "orange" : ""}`}>
                      {fmt(Math.abs(last.shadow - last.primary))}
                    </span>
                  </div>
                </div>
              </div>
            ) : (
              <p className="t-dim hint-line">
                Turn the shadow on. Clients keep getting {primary}; {candidate} gets scored against it on
                every request.
              </p>
            )}
          </div>

          <div className="glass panel shadow-chart">
            <div className="builder-head">
              <span className="eyebrow">modelgate_shadow_divergence_minutes{"{"}primary={primary},shadow={shadow ?? candidate}{"}"}</span>
              <span className={`pill ${on ? "ice" : ""}`}>
                <span className={`dot ${on ? "live" : ""}`} /> n={report.count}
              </span>
            </div>
            <BucketHistogram
              bars={bars}
              height={170}
              color="var(--ice)"
              highlightColor="var(--orange)"
              ariaLabel="Histogram of absolute ETA delta between shadow and primary, by bucket upper bound in minutes"
            />
            <BucketLabels labels={bars.map((b) => b.label)} />
            <p className="hint-line">
              Bucket upper bounds in minutes. Orange buckets are beyond the {report.threshold_minutes} min
              threshold.
            </p>
          </div>

          <motion.dl
            className="shadow-stats"
            initial={false}
            animate={{ opacity: on ? 1 : 0.55 }}
            transition={reduce ? { duration: 0 } : { duration: 0.3 }}
          >
            {[
              ["mean |d|", fmt(report.abs_delta_minutes.mean), "min"],
              ["p50 |d|", fmt(report.abs_delta_minutes.p50), "min"],
              ["p95 |d|", fmt(report.abs_delta_minutes.p95), "min"],
              ["max |d|", fmt(report.abs_delta_minutes.max), "min"],
              [`beyond ${report.threshold_minutes} min`, fmt(report.share_beyond_threshold * 100, 1), "%"],
              ["bias (shadow - primary)", fmt(report.shadow_bias_minutes), "min"],
              ["rel delta p95", fmt(report.rel_delta.p95 * 100, 1), "%"],
              ["window / errors", `${report.window} / ${report.errors}`, ""],
            ].map(([k, v, u]) => (
              <div className="stat glass panel-sm" key={k}>
                <dt className="k">{k}</dt>
                <dd className="v mono">
                  {v}
                  {u && <span className="unit">{u}</span>}
                </dd>
              </div>
            ))}
          </motion.dl>
        </div>
        <p className="hint-line container-note">
          The real run reported n=1000, mean |d| 2.29, p95 |d| 6.35, and 40.1% beyond 2 min on the same
          trip distribution. v2 is the better model (MAE 1.89 vs 3.11), so a wide divergence is the
          candidate correcting the primary, which is exactly what a shadow report is for.
        </p>
      </div>
    </section>
  );
}
