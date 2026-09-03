import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { useState } from "react";
import type { SwapRecord } from "../sim/registry";
import { useService } from "../state/ServiceProvider";
import { SplitBars, Sparkline } from "./charts/Bars";

const COLORS: Record<string, string> = { v1: "var(--orange)", v2: "var(--ice)" };
const WINDOW_S = 24;

function us(ms: number): string {
  return `${Math.round(ms * 1000)} us`;
}

/** performance.now is coarsened to 100 us in a normal browsing context. */
function swapUs(micros: number): string {
  return micros < 100 ? "under 100 us" : `${micros} us`;
}

export function ZeroDropSwap() {
  const { service, gen, refresh } = useService();
  const reduce = useReducedMotion();
  const [busy, setBusy] = useState(false);
  const [flash, setFlash] = useState<SwapRecord | null>(null);

  const primary = service.registry.primary?.version ?? "v1";
  const candidate = service.registry.availableVersions().find((v) => v !== primary) ?? "v2";
  const canRollback = service.registry.previous !== null;
  const stats = gen.stats();
  const columns = gen.recentSplit(WINDOW_S);
  const startedAt = service.now - gen.elapsed;
  const marks = service.registry.swapHistory
    .filter((r) => r.at >= startedAt)
    .map((r) => ({ second: Math.floor(r.at - startedAt), label: `${r.from ?? "boot"} to ${r.to}` }));
  const serverDropped = service.metrics.droppedRequests.get();

  const promote = async () => {
    setBusy(true);
    try {
      const rec = await service.promote(candidate);
      setFlash(rec);
    } finally {
      setBusy(false);
      refresh();
    }
  };

  const rollback = () => {
    const rec = service.rollback();
    setFlash(rec);
    refresh();
  };

  const toggleStream = () => {
    if (gen.running) {
      gen.running = false;
      gen.drain();
    } else {
      gen.running = true;
    }
    refresh();
  };

  const resetRun = () => {
    gen.reset();
    refresh();
  };

  const fill = ((gen.rps - 50) / (600 - 50)) * 100;

  return (
    <section className="section" id="swap" aria-labelledby="swap-title">
      <div className="container">
        <div className="section-head">
          <span className="section-index">03 / zero-drop swap</span>
          <h2 className="section-title" id="swap-title">
            Promote mid-stream. <span className="hero-accent">Watch the counter hold.</span>
          </h2>
          <p className="section-lede">
            The candidate is built and warmed off the request path, then the primary reference is replaced
            in one assignment. Every request captured its model at admission, so anything in flight
            finishes on the version it started with. Dropped means any non-2xx or no response, the same
            accounting as the real load test.
          </p>
        </div>

        <div className="swap-grid">
          <div className="glass panel orange-edge swap-controls">
            <div className="control-row">
              <button type="button" className={`btn ${gen.running ? "" : "btn-primary"}`} onClick={toggleStream}>
                <span className={`dot ${gen.running ? "live" : ""}`} aria-hidden="true" />
                {gen.running ? "Stop stream" : "Start stream"}
              </button>
              <button type="button" className="btn btn-ghost btn-sm" onClick={resetRun}>
                Reset counters
              </button>
            </div>
            <div className="field range-field">
              <label htmlFor="rps">
                request rate <span className="mono t-orange">{gen.rps} rps</span>
              </label>
              <input
                id="rps"
                type="range"
                min={50}
                max={600}
                step={10}
                value={gen.rps}
                style={{ "--fill": `${fill}%` } as React.CSSProperties}
                onChange={(e) => {
                  gen.rps = Number(e.target.value);
                  refresh();
                }}
              />
              <span className="hint">open loop: sends on schedule regardless of responses</span>
            </div>
            <div className="control-row">
              <button
                type="button"
                className={`btn ${candidate === "v2" ? "btn-ice" : "btn-primary"}`}
                onClick={promote}
                disabled={busy}
              >
                {busy ? "Loading and warming" : `Promote ${candidate}`}
              </button>
              <button type="button" className="btn" onClick={rollback} disabled={!canRollback}>
                Rollback
              </button>
            </div>
            <div className="toggle-row">
              <button
                type="button"
                role="switch"
                aria-checked={gen.chaos > 0}
                className="switch orange"
                id="chaos-switch"
                onClick={() => {
                  gen.chaos = gen.chaos > 0 ? 0 : 0.05;
                  refresh();
                }}
              />
              <label htmlFor="chaos-switch" className="toggle-label">
                Inject 5% malformed inputs
                <span className="hint">rejected with 422, counted separately, never a drop</span>
              </label>
            </div>
            <AnimatePresence>
              {flash && (
                <motion.div
                  key={`${flash.kind}-${flash.at}`}
                  className={`swap-flash ${flash.to === "v2" ? "ice" : "orange"}`}
                  role="status"
                  initial={reduce ? false : { opacity: 0, scale: 0.96 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={reduce ? undefined : { opacity: 0 }}
                  transition={{ duration: 0.25 }}
                >
                  <span className="mono">
                    {flash.kind} {flash.from ?? "none"} {"->"} {flash.to}
                  </span>
                  <span className="mono">
                    swap {swapUs(flash.swap_micros)}{flash.load_ms ? `, load+warm ${flash.load_ms.toFixed(2)} ms` : ""}
                  </span>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          <div className="glass panel dropped-tile" aria-live="polite">
            <span className="k eyebrow">dropped requests</span>
            <motion.span
              className={`dropped-value mono ${stats.dropped === 0 && serverDropped === 0 ? "ok" : "bad"}`}
              key={stats.dropped}
              initial={false}
              animate={reduce ? {} : { scale: [1, 1.04, 1] }}
              transition={{ duration: 0.3 }}
            >
              {stats.dropped}
            </motion.span>
            <span className="sub mono">
              server counter {serverDropped} / {stats.sent.toLocaleString()} sent, {stats.ok.toLocaleString()} ok
            </span>
            <span className="sub mono t-dim">
              {stats.rejected} rejected (422) / {gen.inFlight} in flight
            </span>
          </div>

          <div className="glass panel split-panel">
            <div className="builder-head">
              <span className="eyebrow">per-second version split, last {WINDOW_S}s</span>
              <span className="legend mono">
                <i style={{ background: COLORS.v1 }} /> v1
                <i style={{ background: COLORS.v2 }} /> v2
                <i style={{ background: "var(--bad)" }} /> dropped
              </span>
            </div>
            <SplitBars
              columns={columns}
              colors={COLORS}
              order={["v1", "v2"]}
              marks={marks}
              height={170}
              ariaLabel="Stacked requests per second by serving model version, with swap markers"
            />
            <div className="split-foot mono t-dim">
              <span>t+{Math.max(0, Math.floor(gen.elapsed) - WINDOW_S + 1)}s</span>
              <span>
                achieved {stats.achievedRps.toFixed(1)} rps
                {marks.length ? ` / ${marks.length} swap${marks.length > 1 ? "s" : ""} in window` : ""}
              </span>
              <span>t+{Math.floor(gen.elapsed)}s</span>
            </div>
          </div>

          <div className="glass panel latency-panel">
            <div className="builder-head">
              <span className="eyebrow">in-browser compute per request</span>
              <span className="mono t-dim">
                p50 {us(stats.latency.p50)} / p95 {us(stats.latency.p95)} / p99 {us(stats.latency.p99)}
              </span>
            </div>
            <Sparkline
              values={gen.sparklineValues()}
              height={90}
              ariaLabel="Mean compute time per request for each frame, most recent on the right"
            />
            <p className="hint-line">
              Validation, encoding, primary inference, shadow inference, and metrics, measured with
              performance.now around each request. The real service saw p50 1.78 ms end to end over HTTP.
            </p>
          </div>

          <div className="glass panel swap-log">
            <span className="eyebrow">swap history</span>
            <ol className="mono">
              {service.registry.swapHistory
                .slice()
                .reverse()
                .slice(0, 6)
                .map((r, i) => (
                  <li key={`${r.at}-${i}`}>
                    <span className={`pill ${r.kind === "promote" ? "orange" : ""}`}>{r.kind}</span>
                    <span>
                      {r.from ?? "none"} {"->"} {r.to}
                    </span>
                    <span className="t-dim">t+{r.at.toFixed(3)}s</span>
                    <span className="t-dim">{swapUs(r.swap_micros)}</span>
                  </li>
                ))}
            </ol>
          </div>
        </div>
      </div>
    </section>
  );
}
