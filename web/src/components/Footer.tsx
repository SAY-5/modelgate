import { MANIFEST } from "../sim/manifest";
import { WEIGHTS } from "../sim/net";

const REPO = "https://github.com/SAY-5/modelgate";

function paramCount(v: string): number {
  return WEIGHTS[v].layers.reduce((n, l) => n + l.w.length * l.w[0].length + l.b.length, 0);
}

export function Footer() {
  return (
    <footer className="footer" aria-labelledby="footer-title">
      <div className="container footer-grid">
        <div>
          <h2 id="footer-title" className="footer-title">
            What this page is
          </h2>
          <p className="footer-copy">
            A browser port of the ModelGate serving layer. The feature encoding, the strict input rules
            and their rejection reasons, the registry with load-and-warm and a single-assignment swap, the
            shadow tracker, the metric names and buckets, and the open-loop load generator are all ported
            line for line from the Python service to TypeScript in <code>web/src/sim</code>. The two
            models run on the real weights exported from <code>artifacts/eta_v1.pt</code> and{" "}
            <code>eta_v2.pt</code> and match the torch predictions to four decimals; the console
            self-check on page load verifies that, the rejection rules, and a zero-drop burst.
          </p>
          <p className="footer-copy">
            What differs: there is no HTTP here, so latency is in-browser compute rather than the
            1.78 ms p50 the real service measured end to end. Trips are drawn from the same synthetic
            distribution as the real load test with a seeded generator, so every run of this page is
            reproducible.
          </p>
        </div>
        <dl className="footer-facts mono">
          <div>
            <dt>v1</dt>
            <dd>
              MLP 19-64-64-1, {paramCount("v1").toLocaleString()} params, MAE{" "}
              {MANIFEST.versions.v1.metrics.test_mae_minutes} min
            </dd>
          </div>
          <div>
            <dt>v2</dt>
            <dd>
              MLP 19-96-96-96-1, {paramCount("v2").toLocaleString()} params, MAE{" "}
              {MANIFEST.versions.v2.metrics.test_mae_minutes} min
            </dd>
          </div>
          <div>
            <dt>baselines</dt>
            <dd>
              mean {MANIFEST.baselines.mean_predictor_mae}, distance-only linear{" "}
              {MANIFEST.baselines.distance_linear_mae}
            </dd>
          </div>
          <div>
            <dt>source</dt>
            <dd>
              <a href={REPO} target="_blank" rel="noreferrer">
                github.com/SAY-5/modelgate
              </a>
            </dd>
          </div>
          <div>
            <dt>mechanism</dt>
            <dd>
              <a href={`${REPO}/blob/main/ARCHITECTURE.md`} target="_blank" rel="noreferrer">
                ARCHITECTURE.md
              </a>
            </dd>
          </div>
        </dl>
      </div>
    </footer>
  );
}
