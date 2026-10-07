import React, { useLayoutEffect, useRef, useState } from 'react';

// Fixed series order and colour slots (validated categorical palette, see styles.css).
export const SERIES = [
  { key: 'f1', label: 'F1-score', cls: 's-f1', shape: 'circle' },
  { key: 'precision', label: 'Precision', cls: 's-precision', shape: 'square' },
  { key: 'recall', label: 'Recall', cls: 's-recall', shape: 'diamond' },
];

const M = { top: 34, right: 112, bottom: 40, left: 52 };
const HEIGHT = 300;
const pct = (v) => (v == null ? '—' : `${(v * 100).toFixed(1)}%`);

export function Marker({ shape, x, y, r = 4.5, className }) {
  if (shape === 'square') return <rect className={className} x={x - r} y={y - r} width={r * 2} height={r * 2} rx={1.5} />;
  if (shape === 'diamond') {
    const d = r * 1.3;
    return <path className={className} d={`M${x},${y - d}L${x + d},${y}L${x},${y + d}L${x - d},${y}Z`} />;
  }
  return <circle className={className} cx={x} cy={y} r={r} />;
}

export function LegendKey({ s }) {
  return (
    <svg width="22" height="12" aria-hidden="true" className={`legend-key ${s.cls}`}>
      <line x1="1" x2="21" y1="6" y2="6" className="line" />
      <Marker shape={s.shape} x={11} y={6} r={3.5} className="mark" />
    </svg>
  );
}

function niceDomain(values) {
  const lo = Math.min(...values);
  const span = 1 - lo;
  const step = span <= 0.15 ? 0.025 : span <= 0.3 ? 0.05 : span <= 0.6 ? 0.1 : 0.2;
  const min = Math.max(0, Math.floor((lo - step / 2) / step) * step);
  const ticks = [];
  for (let t = min; t <= 1 + 1e-9; t += step) ticks.push(Math.round(t * 1000) / 1000);
  return { min, ticks };
}

/** Precision, recall and F1 against confidence threshold; best threshold highlighted. */
export default function MetricsChart({ rows, bestIndex, digits }) {
  const wrapRef = useRef(null);
  const [width, setWidth] = useState(640);
  const [hover, setHover] = useState(null);

  useLayoutEffect(() => {
    const el = wrapRef.current;
    if (!el) return undefined;
    const ro = new ResizeObserver(([e]) => setWidth(Math.max(320, Math.round(e.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const narrow = width < 520;
  const m = { ...M, right: narrow ? 16 : M.right };
  const iw = width - m.left - m.right;
  const ih = HEIGHT - m.top - m.bottom;
  const n = rows.length;
  const values = rows.flatMap((r) => SERIES.map((s) => r[s.key])).filter((v) => v != null);
  const { min, ticks } = niceDomain(values.length ? values : [0]);
  const x = (i) => m.left + (n === 1 ? iw / 2 : (i / (n - 1)) * iw);
  const y = (v) => m.top + (1 - (v - min) / (1 - min)) * ih;
  const colW = n === 1 ? iw : iw / (n - 1);
  const showMarks = n <= 25;
  const labelEvery = Math.max(1, Math.ceil(n / Math.max(2, Math.floor(iw / 56))));

  // direct labels at the right end, nudged apart so they never collide
  const last = rows[n - 1];
  const ends = SERIES.filter((s) => last?.[s.key] != null)
    .map((s) => ({ s, y: y(last[s.key]) }))
    .sort((a, b) => a.y - b.y);
  for (let i = 1; i < ends.length; i++) ends[i].y = Math.max(ends[i].y, ends[i - 1].y + 15);

  const path = (key) => {
    let d = '';
    let pen = false;
    rows.forEach((r, i) => {
      if (r[key] == null) { pen = false; return; }
      d += `${pen ? 'L' : 'M'}${x(i)},${y(r[key])}`;
      pen = true;
    });
    return d;
  };

  const best = bestIndex != null ? rows[bestIndex] : null;
  const hv = hover != null ? rows[hover] : null;
  const tipLeft = hover != null ? Math.min(Math.max(x(hover) + 14, 8), width - 200) : 0;
  const tipFlip = hover != null && x(hover) + 214 > width;

  return (
    <div className="metrics-chart" ref={wrapRef}>
      <ul className="chart-legend" aria-hidden="true">
        {SERIES.map((s) => <li key={s.key}><LegendKey s={s} />{s.label}</li>)}
        {best && <li className="legend-best"><span className="best-swatch" />Selected threshold</li>}
      </ul>
      <svg
        width={width}
        height={HEIGHT}
        role="img"
        aria-label={`Precision, recall and F1-score for confidence thresholds ${rows[0]?.threshold.toFixed(digits)} to ${last?.threshold.toFixed(digits)}.${best ? ` Selected threshold ${best.threshold.toFixed(digits)} with F1-score ${pct(best.f1)}.` : ''} The table below lists every value.`}
        onMouseLeave={() => setHover(null)}
      >
        {/* best-threshold band */}
        {best && (
          <g className="best-band">
            <rect x={x(bestIndex) - Math.min(colW / 2, 22)} y={m.top - 22} width={Math.min(colW, 44)} height={ih + 22} rx={8} />
            <text x={x(bestIndex)} y={m.top - 8} textAnchor="middle">Best</text>
          </g>
        )}

        {/* grid + y axis */}
        {ticks.map((t) => (
          <g key={t} className="grid-line">
            <line x1={m.left} x2={m.left + iw} y1={y(t)} y2={y(t)} />
            <text x={m.left - 10} y={y(t)} dy="0.32em" textAnchor="end">{Math.round(t * 1000) / 10}%</text>
          </g>
        ))}

        {/* x axis */}
        <line className="axis-line" x1={m.left} x2={m.left + iw} y1={m.top + ih} y2={m.top + ih} />
        {rows.map((r, i) => (i % labelEvery === 0 || i === n - 1 || i === bestIndex) && (
          <text key={r.threshold} className={`x-tick ${i === bestIndex ? 'best' : ''}`} x={x(i)} y={m.top + ih + 20} textAnchor="middle">
            {r.threshold.toFixed(digits)}
          </text>
        ))}
        <text className="axis-title" x={m.left + iw / 2} y={HEIGHT - 2} textAnchor="middle">Confidence threshold</text>

        {/* crosshair */}
        {hv && <line className="crosshair" x1={x(hover)} x2={x(hover)} y1={m.top} y2={m.top + ih} />}

        {/* series: 2px lines, ≥8px markers with a surface ring */}
        {SERIES.map((s) => (
          <g key={s.key} className={`series ${s.cls}`}>
            <path className="line" d={path(s.key)} />
            {rows.map((r, i) => r[s.key] != null && (showMarks || i === hover || i === bestIndex) && (
              <Marker key={i} shape={s.shape} x={x(i)} y={y(r[s.key])} r={i === hover || i === bestIndex ? 5.5 : 4.5} className="mark" />
            ))}
          </g>
        ))}

        {/* direct labels */}
        {!narrow && ends.map(({ s, y: ly }) => (
          <g key={s.key} className={`end-label ${s.cls}`} transform={`translate(${m.left + iw + 12},${ly})`}>
            <Marker shape={s.shape} x={4} y={0} r={3.5} className="mark" />
            <text x={14} dy="0.32em">{s.label}</text>
          </g>
        ))}

        {/* hit columns, larger than the marks */}
        {rows.map((r, i) => (
          <rect key={r.threshold} className="hit" x={x(i) - colW / 2} y={m.top} width={colW} height={ih} onMouseEnter={() => setHover(i)} />
        ))}
      </svg>

      {hv && (
        <div className="tooltip floating chart-tip" role="tooltip"
             style={{ transform: `translate(${tipFlip ? x(hover) - 214 : tipLeft}px, ${m.top + 30}px)` }}>
          <span className="tt-title">
            Threshold {hv.threshold.toFixed(digits)}
            {hover === bestIndex && <span className="tt-best">Best</span>}
          </span>
          {SERIES.map((s) => (
            <span key={s.key} className="tt-row"><span><LegendKey s={s} /> {s.label}</span><strong>{pct(hv[s.key])}</strong></span>
          ))}
          <span className="tt-row"><span>Detections</span><strong>{hv.detections.toLocaleString()}</strong></span>
        </div>
      )}
    </div>
  );
}
