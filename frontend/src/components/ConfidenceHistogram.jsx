import React, { useState } from 'react';

const BINS = [
  { lo: 0, hi: 0.8, label: '<80' },
  { lo: 0.8, hi: 0.85, label: '80–85' },
  { lo: 0.85, hi: 0.9, label: '85–90' },
  { lo: 0.9, hi: 0.95, label: '90–95' },
  { lo: 0.95, hi: 1.0001, label: '95–100' },
];

/** Distribution of detection confidences — single series, direct-labelled bars. */
export default function ConfidenceHistogram({ confidences }) {
  const [hover, setHover] = useState(null);
  const counts = BINS.map((b) => confidences.filter((c) => c >= b.lo && c < b.hi).length);
  const max = Math.max(1, ...counts);
  const total = confidences.length;

  return (
    <div className="histogram" role="table" aria-label="Detections by confidence range">
      <div className="hist-bars" role="row">
        {BINS.map((b, i) => (
          <div
            key={b.label}
            role="cell"
            className={`hist-col ${hover != null && hover !== i ? 'dim' : ''}`}
            onMouseEnter={() => setHover(i)}
            onMouseLeave={() => setHover(null)}
            aria-label={`${b.label}%: ${counts[i]} detections`}
          >
            <span className="hist-val">{counts[i] || ''}</span>
            <span className="hist-bar" style={{ height: `${(counts[i] / max) * 100}%` }} />
            {hover === i && (
              <span className="tooltip" role="tooltip">
                <span className="tt-title">Confidence {b.label}%</span>
                <span className="tt-row"><span>Detections</span><strong>{counts[i]}</strong></span>
                <span className="tt-row"><span>Share</span><strong>{total ? Math.round((counts[i] / total) * 100) : 0}%</strong></span>
              </span>
            )}
          </div>
        ))}
      </div>
      <div className="hist-axis" aria-hidden="true">
        {BINS.map((b) => <span key={b.label}>{b.label}</span>)}
      </div>
    </div>
  );
}
