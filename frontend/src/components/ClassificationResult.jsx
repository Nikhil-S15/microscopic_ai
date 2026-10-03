// ClassificationResult.jsx
// Step 1 output — species name, confidence, all class probabilities, risk, Step-2 status

import React from 'react';
import { CheckCircle, Loader2 } from 'lucide-react';

const RISK_CONFIG = {
  High:   { color: '#ef4444', bg: '#fef2f2', icon: '⚠️',  label: 'High Risk' },
  Medium: { color: '#f97316', bg: '#fff7ed', icon: '⚡',  label: 'Moderate Risk' },
  Low:    { color: '#22c55e', bg: '#f0fdf4', icon: '✅',  label: 'Low Risk' },
};

export default function ClassificationResult({
  result,
  detecting,
  detectionDone,
  imageName,
}) {
  if (!result) return null;

  const risk    = RISK_CONFIG[result.risk_level] ?? RISK_CONFIG.Medium;
  // Show top-4 classes (sorted descending, already done by backend)
  const topFour = result.all_predictions.slice(0, 4);

  return (
    <div className="classify-card">

      {/* ── Header bar ─────────────────────────────────────────────────── */}
      <div className="classify-header" style={{ borderTopColor: result.color }}>
        <div className="classify-badge">Step 1 — Species Identified</div>

        <div className="classify-main">
          {/* Icon */}
          <div
            className="classify-icon"
            style={{ background: result.color + '20', color: result.color }}
          >
            🦠
          </div>

          {/* Name + meta tags */}
          <div className="classify-title-block">
            <h2 className="classify-species" style={{ color: result.color }}>
              {result.predicted_class}
            </h2>
            <div className="classify-meta">
              <span className="meta-tag">{result.gram_type}</span>
              <span className="meta-tag">{result.shape}</span>
              <span
                className="meta-tag risk"
                style={{ background: risk.bg, color: risk.color }}
              >
                {risk.icon} {risk.label}
              </span>
            </div>
          </div>

          {/* Big confidence number */}
          {/* <div className="classify-confidence-big">
            <div className="conf-pct" style={{ color: result.color }}>
              {(result.confidence * 100).toFixed(1)}%
            </div>
            <div className="conf-label">Confidence</div>
          </div> */}
        </div>

        {/* Confidence bar */}
        <div className="classify-conf-bar-wrap">
          <div
            className="classify-conf-bar"
            style={{
              width: `${result.confidence * 100}%`,
              background: result.color,
            }}
          />
        </div>
      </div>

      {/* ── All class probabilities ─────────────────────────────────────── */}
      {/* <div className="classify-probs">
        <h4 className="probs-title">All Class Probabilities</h4>
        <div className="probs-list">
          {topFour.map((pred, i) => (
            <div
              key={pred.class_id}
              className={`prob-row ${i === 0 ? 'prob-top' : ''}`}
            >
              <span className="prob-name">{pred.class_name}</span>
              <div className="prob-bar-wrap">
                <div
                  className="prob-bar-fill"
                  style={{
                    width: `${pred.probability * 100}%`,
                    background: i === 0 ? pred.color : pred.color + '70',
                  }}
                />
              </div>
              <span
                className="prob-pct"
                style={{ color: i === 0 ? pred.color : '#64748b' }}
              >
                {(pred.probability * 100).toFixed(1)}%
              </span>
            </div>
          ))}
        </div>
      </div> */}

      {/* ── Step 2 status strip ─────────────────────────────────────────── */}
      {(detecting || detectionDone) && (
        <div className={`classify-step2-status ${detectionDone ? 'done' : 'loading'}`}>
          {detecting && (
            <>
              <Loader2 size={16} className="spin" />
              <span>Step 2 running — locating individual cells with bounding boxes…</span>
            </>
          )}
          {detectionDone && (
            <>
              <CheckCircle size={16} />
              <span>Step 2 complete — scroll down to see bounding box detections</span>
            </>
          )}
        </div>
      )}

      <style>{`
        .classify-card {
          background: #fff;
          border-radius: 20px;
          box-shadow: 0 4px 24px rgba(0,0,0,0.08);
          overflow: hidden;
          margin-bottom: 24px;
          font-family: 'DM Sans', 'Segoe UI', sans-serif;
        }

        .classify-header {
          padding: 28px 28px 0;
          border-top: 4px solid #3b82f6;
        }

        .classify-badge {
          display: inline-block;
          background: #f1f5f9;
          color: #475569;
          font-size: 0.72rem;
          font-weight: 700;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          padding: 4px 12px;
          border-radius: 20px;
          margin-bottom: 16px;
        }

        .classify-main {
          display: flex;
          align-items: center;
          gap: 20px;
          margin-bottom: 20px;
        }

        .classify-icon {
          width: 64px;
          height: 64px;
          border-radius: 16px;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 2rem;
          flex-shrink: 0;
        }

        .classify-title-block { flex: 1; }

        .classify-species {
          font-size: 1.8rem;
          font-weight: 800;
          margin: 0 0 8px;
          font-style: italic;
          letter-spacing: -0.02em;
        }

        .classify-meta {
          display: flex;
          flex-wrap: wrap;
          gap: 8px;
        }

        .meta-tag {
          background: #f1f5f9;
          color: #475569;
          padding: 4px 12px;
          border-radius: 20px;
          font-size: 0.78rem;
          font-weight: 600;
        }

        .meta-tag.risk { font-weight: 700; }

        .classify-confidence-big {
          text-align: center;
          flex-shrink: 0;
        }

        .conf-pct {
          font-size: 2.4rem;
          font-weight: 900;
          line-height: 1;
        }

        .conf-label {
          font-size: 0.72rem;
          font-weight: 600;
          color: #94a3b8;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          margin-top: 4px;
        }

        .classify-conf-bar-wrap {
          height: 6px;
          background: #f1f5f9;
          border-radius: 99px;
          margin-bottom: 24px;
          overflow: hidden;
        }

        .classify-conf-bar {
          height: 100%;
          border-radius: 99px;
          transition: width 0.8s cubic-bezier(0.34, 1.56, 0.64, 1);
        }

        .classify-probs {
          padding: 20px 28px;
          border-top: 1px solid #f1f5f9;
        }

        .probs-title {
          font-size: 0.8rem;
          font-weight: 700;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          color: #94a3b8;
          margin: 0 0 14px;
        }

        .probs-list {
          display: flex;
          flex-direction: column;
          gap: 10px;
        }

        .prob-row {
          display: grid;
          grid-template-columns: 200px 1fr 56px;
          align-items: center;
          gap: 12px;
        }

        .prob-row.prob-top .prob-name {
          font-weight: 700;
          color: #1e293b;
        }

        .prob-name {
          font-size: 0.85rem;
          font-weight: 500;
          color: #64748b;
          font-style: italic;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }

        .prob-bar-wrap {
          height: 10px;
          background: #f1f5f9;
          border-radius: 99px;
          overflow: hidden;
        }

        .prob-bar-fill {
          height: 100%;
          border-radius: 99px;
          transition: width 0.7s ease;
        }

        .prob-pct {
          font-size: 0.82rem;
          font-weight: 700;
          text-align: right;
        }

        .classify-step2-status {
          display: flex;
          align-items: center;
          gap: 10px;
          padding: 14px 28px;
          font-size: 0.85rem;
          font-weight: 600;
        }

        .classify-step2-status.loading {
          background: #eff6ff;
          color: #2563eb;
        }

        .classify-step2-status.done {
          background: #f0fdf4;
          color: #16a34a;
        }

        .spin {
          animation: spin 1s linear infinite;
          flex-shrink: 0;
        }

        @keyframes spin {
          from { transform: rotate(0deg); }
          to   { transform: rotate(360deg); }
        }

        @media (max-width: 640px) {
          .classify-main { flex-wrap: wrap; }
          .prob-row { grid-template-columns: 130px 1fr 50px; }
          .classify-species { font-size: 1.4rem; }
        }
      `}</style>
    </div>
  );
}
