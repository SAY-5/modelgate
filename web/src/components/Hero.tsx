import { motion, useReducedMotion } from "framer-motion";
import { useCountUp } from "../hooks/useCountUp";
import { REAL_DEMO } from "../sim/manifest";
import { useService } from "../state/ServiceProvider";

const REPO = "https://github.com/SAY-5/modelgate";

/** Per-second split from the committed `make demo` run: 20 columns. */
const DEMO_SPLIT = Array.from({ length: 20 }, (_, s) => {
  if (s < 10) return { v1: 200, v2: 0 };
  if (s === 10) return { v1: 1, v2: 199 };
  return { v1: 0, v2: 200 };
});

export function Hero() {
  const reduce = useReducedMotion();
  const { service } = useService();
  const total = useCountUp(REAL_DEMO.total_requests, 1800, 500);
  const ok = useCountUp(REAL_DEMO.successes, 1800, 650);
  const bootSwap = service.registry.swapHistory[0];
  const swapMs = Math.round((REAL_DEMO.swap_completed_at_s - REAL_DEMO.swap_requested_at_s) * 1000);
  const settled = total === REAL_DEMO.total_requests;

  const rise = (delay: number) =>
    reduce
      ? {}
      : {
          initial: { opacity: 0, y: 18 },
          animate: { opacity: 1, y: 0 },
          transition: { duration: 0.7, delay, ease: [0.22, 1, 0.36, 1] as const },
        };

  return (
    <section className="hero" id="top" aria-labelledby="hero-title">
      <div className="container hero-grid">
        <div className="hero-copy">
          <motion.p className="eyebrow" {...rise(0.05)}>
            ModelGate / browser port of the serving layer
          </motion.p>
          <motion.h1 id="hero-title" className="hero-title" {...rise(0.15)}>
            Swap the model.
            <br />
            <span className="hero-accent">Drop nothing.</span>
          </motion.h1>
          <motion.p className="hero-lede" {...rise(0.3)}>
            A PyTorch ETA model behind strict input checks, a shadow slot for the candidate version,
            Prometheus metrics, and a primary swap that is one reference assignment. This page runs
            the same mechanism in TypeScript on the real v1 and v2 weights, so every number below is
            computed, not staged.
          </motion.p>
          <motion.div className="hero-cta" {...rise(0.42)}>
            <a href="#swap" className="btn btn-primary">
              Run the swap under load
            </a>
            <a href="#checks" className="btn">
              Build a request
            </a>
            <a href={REPO} className="btn btn-ghost" target="_blank" rel="noreferrer">
              Source
            </a>
          </motion.div>
          <motion.dl className="hero-stats" {...rise(0.55)}>
            <div className="stat">
              <dt className="k">requests / successes</dt>
              <dd className="v orange mono">
                {total.toLocaleString()} / {ok.toLocaleString()}
              </dd>
              <dd className="sub">200 rps for 20 s, open loop</dd>
            </div>
            <div className="stat">
              <dt className="k">dropped</dt>
              <dd className={`v mono ${settled ? "ok" : ""}`}>0</dd>
              <dd className="sub">client count and server counter</dd>
            </div>
            <div className="stat">
              <dt className="k">promote round trip</dt>
              <dd className="v mono">{swapMs} ms</dd>
              <dd className="sub">
                t+{REAL_DEMO.swap_requested_at_s.toFixed(3)}s to t+{REAL_DEMO.swap_completed_at_s.toFixed(3)}s
              </dd>
            </div>
            <div className="stat">
              <dt className="k">swap critical section</dt>
              <dd className="v ice mono">{bootSwap ? (bootSwap.swap_micros < 100 ? "< 100 us" : `${bootSwap.swap_micros} us`) : "n/a"}</dd>
              <dd className="sub">measured here, at boot, this tab</dd>
            </div>
          </motion.dl>
        </div>

        <motion.div
          className="glass orange-edge terminal"
          role="figure"
          aria-label="Output of make demo from the repository README"
          {...(reduce
            ? {}
            : {
                initial: { opacity: 0, y: 24, rotate: -0.6 },
                animate: { opacity: 1, y: 0, rotate: 0 },
                transition: { duration: 0.9, delay: 0.35, ease: [0.22, 1, 0.36, 1] },
              })}
        >
          <div className="terminal-bar">
            <span className="terminal-dots" aria-hidden="true">
              <i />
              <i />
              <i />
            </span>
            <span className="mono">make demo</span>
            <span className="pill ok">
              <span className="dot" /> PASS
            </span>
          </div>
          <pre className="terminal-body mono">
            <span className="t-dim">ModelGate load test: version swap under load</span>
            {"\n"}target rate        <b>{REAL_DEMO.target_rps}.0 rps</b> for {REAL_DEMO.duration_s}.0 s
            {"\n"}total requests     <b>{total}</b>
            {"\n"}successes (2xx)    <b>{ok}</b>
            {"\n"}dropped requests   <b className="t-ok">{REAL_DEMO.dropped}</b>  (non-2xx or no response)
            {"\n"}server dropped ctr <b className="t-ok">{REAL_DEMO.server_dropped_metric}.0</b>
            {"\n"}latency ms         p50 {REAL_DEMO.latency_ms.p50}  p95 {REAL_DEMO.latency_ms.p95}  p99 {REAL_DEMO.latency_ms.p99}
            {"\n"}shadow enabled     t+{REAL_DEMO.shadow_enabled_at_s}s (v2 shadowing)
            {"\n"}swap v1 -&gt; v2     requested t+{REAL_DEMO.swap_requested_at_s.toFixed(1)}s, completed{" "}
            <b className="t-orange">t+{REAL_DEMO.swap_completed_at_s}s</b>
            {"\n"}versions before    {"{"}v1: 2001, v2: 1{"}"}
            {"\n"}versions after     {"{"}v2: 1998{"}"}
            {"\n"}shadow report      n=1000 mean|d|={REAL_DEMO.shadow_report.mean_abs} p95|d|=
            {REAL_DEMO.shadow_report.p95_abs} beyond 2.0min={REAL_DEMO.shadow_report.beyond_2min}
          </pre>
          <div className="terminal-split" aria-label="Per-second version split of the real run">
            {DEMO_SPLIT.map((c, s) => (
              <motion.div
                key={s}
                className="split-col"
                title={`${s}s: v1 ${c.v1}, v2 ${c.v2}`}
                initial={reduce ? false : { scaleY: 0 }}
                animate={{ scaleY: 1 }}
                transition={reduce ? { duration: 0 } : { delay: 0.9 + s * 0.05, duration: 0.5, ease: "easeOut" }}
              >
                <span className="seg v2" style={{ flexGrow: c.v2 }} />
                <span className="seg v1" style={{ flexGrow: c.v1 }} />
              </motion.div>
            ))}
            <span className="split-axis mono">
              <span>0s</span>
              <span className="t-orange">swap at 10s: v1 1, v2 199</span>
              <span>19s</span>
            </span>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
