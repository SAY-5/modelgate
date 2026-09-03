/** Small SVG chart primitives shared by the sections. Pure, data in, marks out. */

import { motion, useReducedMotion } from "framer-motion";

export interface BucketBar {
  label: string;
  value: number;
  highlight?: boolean;
}

/** Vertical histogram with labelled buckets. */
export function BucketHistogram({
  bars,
  height = 150,
  color = "var(--ice)",
  highlightColor = "var(--orange)",
  ariaLabel,
}: {
  bars: BucketBar[];
  height?: number;
  color?: string;
  highlightColor?: string;
  ariaLabel: string;
}) {
  const reduce = useReducedMotion();
  const max = Math.max(1, ...bars.map((b) => b.value));
  const w = 100;
  const gap = 1.2;
  const bw = (w - gap * (bars.length - 1)) / bars.length;
  const plotH = height - 26;
  return (
    <svg
      viewBox={`0 0 ${w} ${height}`}
      preserveAspectRatio="none"
      role="img"
      aria-label={ariaLabel}
      className="chart chart-hist"
      style={{ height }}
    >
      {bars.map((b, i) => {
        const h = Math.max(0, (b.value / max) * plotH);
        const x = i * (bw + gap);
        return (
          <g key={b.label}>
            <rect x={x} y={0} width={bw} height={plotH} fill="rgba(255,255,255,0.03)" />
            <motion.rect
              x={x}
              width={bw}
              fill={b.highlight ? highlightColor : color}
              initial={false}
              animate={{ y: plotH - h, height: h }}
              transition={reduce ? { duration: 0 } : { type: "spring", stiffness: 160, damping: 26 }}
              rx={0.6}
            />
          </g>
        );
      })}
    </svg>
  );
}

export function BucketLabels({ labels }: { labels: string[] }) {
  return (
    <div className="chart-labels mono" aria-hidden="true">
      {labels.map((l) => (
        <span key={l}>{l}</span>
      ))}
    </div>
  );
}

/** Stacked per-second bars: one column per second, segments per version. */
export function SplitBars({
  columns,
  colors,
  order,
  marks = [],
  height = 160,
  ariaLabel,
}: {
  columns: { second: number; counts: Record<string, number>; dropped: number }[];
  colors: Record<string, string>;
  order: string[];
  marks?: { second: number; label: string }[];
  height?: number;
  ariaLabel: string;
}) {
  const reduce = useReducedMotion();
  const max = Math.max(
    1,
    ...columns.map((c) => Object.values(c.counts).reduce((a, b) => a + b, 0) + c.dropped),
  );
  const w = 100;
  const gap = 1;
  const n = Math.max(columns.length, 1);
  const bw = (w - gap * (n - 1)) / n;
  const plotH = height - 4;
  return (
    <svg
      viewBox={`0 0 ${w} ${height}`}
      preserveAspectRatio="none"
      role="img"
      aria-label={ariaLabel}
      className="chart chart-split"
      style={{ height }}
    >
      {columns.map((c, i) => {
        const x = i * (bw + gap);
        let acc = 0;
        const segs = order
          .filter((v) => c.counts[v])
          .map((v) => {
            const h = Math.max(0, (c.counts[v] / max) * plotH);
            const y = plotH - acc - h;
            acc += h;
            return { v, y, h };
          });
        const dropH = (c.dropped / max) * plotH;
        return (
          <g key={c.second}>
            <rect x={x} y={0} width={bw} height={plotH} fill="rgba(255,255,255,0.03)" />
            {segs.map((s) => (
              <motion.rect
                key={s.v}
                x={x}
                width={bw}
                fill={colors[s.v] ?? "var(--ink-3)"}
                initial={false}
                animate={{ y: s.y, height: s.h }}
                transition={reduce ? { duration: 0 } : { duration: 0.16, ease: "linear" }}
              />
            ))}
            {c.dropped > 0 && (
              <rect x={x} y={plotH - acc - dropH} width={bw} height={dropH} fill="var(--bad)" />
            )}
          </g>
        );
      })}
      {marks.map((m) => {
        const idx = columns.findIndex((c) => c.second === m.second);
        if (idx < 0) return null;
        const x = idx * (bw + gap);
        return (
          <g key={`${m.second}-${m.label}`}>
            <line x1={x} x2={x} y1={0} y2={plotH} stroke="var(--ink)" strokeWidth={0.4} strokeDasharray="1 1" />
          </g>
        );
      })}
    </svg>
  );
}

/** Line sparkline with soft fill. */
export function Sparkline({
  values,
  height = 80,
  color = "var(--orange)",
  ariaLabel,
}: {
  values: number[];
  height?: number;
  color?: string;
  ariaLabel: string;
}) {
  const w = 100;
  const n = values.length;
  const max = Math.max(1e-9, ...values);
  const pts = values.map((v, i) => {
    const x = n > 1 ? (i / (n - 1)) * w : 0;
    const y = height - 6 - (v / max) * (height - 12);
    return [x, y] as const;
  });
  const path = pts.length ? "M" + pts.map(([x, y]) => `${x.toFixed(2)} ${y.toFixed(2)}`).join(" L") : "";
  const area = pts.length ? `${path} L${w} ${height} L0 ${height} Z` : "";
  return (
    <svg
      viewBox={`0 0 ${w} ${height}`}
      preserveAspectRatio="none"
      role="img"
      aria-label={ariaLabel}
      className="chart chart-spark"
      style={{ height }}
    >
      <defs>
        <linearGradient id="spark-fill" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0" stopColor={color} stopOpacity="0.35" />
          <stop offset="1" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      {area && <path d={area} fill="url(#spark-fill)" />}
      {path && <path d={path} fill="none" stroke={color} strokeWidth={0.9} vectorEffect="non-scaling-stroke" />}
    </svg>
  );
}
