import React, { useId } from 'react';

/** Single-value radial gauge (0–1). The number in the centre is the headline. */
export default function ConfidenceRing({ value, size = 132, stroke = 10, label = 'Mean confidence' }) {
  const gid = `ring-${useId().replace(/:/g, '')}`;
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const v = value == null ? 0 : Math.max(0, Math.min(1, value));
  return (
    <div className="ring" style={{ width: size, height: size }} role="img" aria-label={`${label}: ${value == null ? 'not available' : `${Math.round(v * 100)} percent`}`}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <defs>
          <linearGradient id={gid} x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stopColor="var(--grad-a)" />
            <stop offset="100%" stopColor="var(--grad-b)" />
          </linearGradient>
        </defs>
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="var(--track)" strokeWidth={stroke} />
        <circle
          cx={size / 2} cy={size / 2} r={r} fill="none" stroke={`url(#${gid})`} strokeWidth={stroke}
          strokeLinecap="round" strokeDasharray={`${c * v} ${c}`} transform={`rotate(-90 ${size / 2} ${size / 2})`}
          className="ring-arc"
        />
      </svg>
      <span className="ring-center">
        <strong>{value == null ? '—' : Math.round(v * 100)}<small>{value == null ? '' : '%'}</small></strong>
        <span>{label}</span>
      </span>
    </div>
  );
}
