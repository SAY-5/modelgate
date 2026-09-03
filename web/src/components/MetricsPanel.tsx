import { useState } from "react";
import type { Histogram, Labels } from "../sim/metrics";
import { useService } from "../state/ServiceProvider";

interface Row {
  series: string;
  value: string;
  tone?: "ok" | "orange" | "ice" | "bad" | "dim";
  note?: string;
}

function lbl(labels: Labels): string {
  const keys = Object.keys(labels);
  return keys.length ? `{${keys.map((k) => `${k}="${labels[k]}"`).join(",")}}` : "";
}

function toneFor(labels: Labels): Row["tone"] {
  const v = labels.version ?? labels.shadow ?? labels.to;
  if (v === "v1") return "orange";
  if (v === "v2") return "ice";
  return undefined;
}

function histRows(h: Histogram, unit: (v: number) => string, extraQ = true): Row[] {
  return h.series().flatMap((c) => {
    const base = `${h.name}${lbl(c.labels)}`;
    const rows: Row[] = [{ series: `${base} count`, value: String(c.count), tone: toneFor(c.labels) }];
    if (extraQ && c.count) {
      rows.push({
        series: `histogram_quantile(0.5 / 0.95 / 0.99), bucket estimate`,
        value: `${unit(h.quantile(0.5, c))} / ${unit(h.quantile(0.95, c))} / ${unit(h.quantile(0.99, c))}`,
        tone: "dim",
      });
    }
    return rows;
  });
}

export function MetricsPanel() {
  const { service } = useService();
  const m = service.metrics;
  const [raw, setRaw] = useState(false);

  const ms = (s: number) => `${(s * 1000).toFixed(3)} ms`;
  const min = (v: number) => `${v.toFixed(2)} min`;

  const groups: { title: string; rows: Row[] }[] = [
    {
      title: "traffic",
      rows: m.requests.series().map((s) => ({
        series: `${m.requests.name}${lbl(s.labels)}`,
        value: s.value.toLocaleString(),
        tone: s.labels.outcome === "error" ? "bad" : toneFor(s.labels),
      })),
    },
    { title: "latency", rows: histRows(m.requestLatency, ms) },
    { title: "eta distribution", rows: histRows(m.predictionsEta, min) },
    {
      title: "input rejections",
      rows: m.inputRejections.series().length
        ? m.inputRejections.series().map((s) => ({
            series: `${m.inputRejections.name}${lbl(s.labels)}`,
            value: String(s.value),
            tone: "bad",
          }))
        : [{ series: m.inputRejections.name, value: "0", tone: "dim", note: "no rejections yet" }],
    },
    {
      title: "shadow",
      rows: [
        ...histRows(m.shadowDivergence, min),
        ...m.shadowRequests.series().map((s) => ({
          series: `${m.shadowRequests.name}${lbl(s.labels)}`,
          value: String(s.value),
          tone: toneFor(s.labels),
        })),
      ],
    },
    {
      title: "versions",
      rows: [
        ...m.modelVersionInfo.series().map((s) => ({
          series: `${m.modelVersionInfo.name}${lbl(s.labels)}`,
          value: String(s.value),
          tone: s.value ? toneFor(s.labels) : ("dim" as const),
        })),
        ...m.versionSwaps.series().map((s) => ({
          series: `${m.versionSwaps.name}${lbl(s.labels)}`,
          value: String(s.value),
        })),
        { series: m.modelsLoaded.name, value: String(m.modelsLoaded.get()) },
      ],
    },
    {
      title: "the one that must stay 0",
      rows: [
        {
          series: m.droppedRequests.name,
          value: String(m.droppedRequests.get()),
          tone: m.droppedRequests.get() === 0 ? "ok" : "bad",
          note: "no labels, counts any /predict that ended in a 5xx",
        },
      ],
    },
  ];

  return (
    <section className="section" id="metrics" aria-labelledby="metrics-title">
      <div className="container">
        <div className="section-head">
          <span className="section-index">04 / metrics</span>
          <h2 className="section-title" id="metrics-title">
            Every series, live, in Prometheus shape.
          </h2>
          <p className="section-lede">
            The same metric names and labels the real service exposes on <code>/metrics</code>, updated
            from the in-browser stream. Per-version labels make a swap visible as one series ending and
            another starting. Quantiles use the histogram_quantile estimator over the same buckets.
          </p>
        </div>

        <div className="metrics-grid">
          {groups.map((g) => (
            <div className="glass panel metrics-group" key={g.title}>
              <span className="eyebrow">{g.title}</span>
              <table className="metrics-table mono">
                <tbody>
                  {g.rows.map((r, i) => (
                    <tr key={`${r.series}-${i}`}>
                      <th scope="row">
                        {r.series}
                        {r.note && <span className="t-dim note">{r.note}</span>}
                      </th>
                      <td className={r.tone ? `t-${r.tone}` : ""}>{r.value}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ))}
        </div>

        <div className="glass panel exposition">
          <button
            type="button"
            className="btn btn-sm"
            aria-expanded={raw}
            aria-controls="exposition-text"
            onClick={() => setRaw((v) => !v)}
          >
            {raw ? "Hide" : "Show"} raw GET /metrics exposition
          </button>
          {raw && (
            <pre id="exposition-text" className="mono exposition-text" tabIndex={0}>
              {service.exposition()}
            </pre>
          )}
        </div>
      </div>
    </section>
  );
}
