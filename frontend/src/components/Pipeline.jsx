import React from 'react';
import { Check, Filter, ImageUp, ScanLine, ShieldCheck } from 'lucide-react';

/**
 * Detection process: where the batch is in the pipeline, plus live progress of
 * the sample currently being scanned.
 */
export default function Pipeline({ items, running, noiseGate, avgTime }) {
  const queued = items.filter((i) => i.status === 'queued').length;
  const active = items.find((i) => i.status === 'analyzing');
  const finished = items.filter((i) => i.status === 'done' || i.status === 'error').length;
  const allDone = !active && queued === 0;

  // step index currently in progress (0-based); 4 = complete
  const step = allDone ? 4 : active ? (active.progress > 0 ? 2 : 1) : 0;
  const steps = [
    { icon: ImageUp, title: 'Sample intake', sub: `${items.length} received` },
    { icon: Filter, title: 'Pre-processing', sub: noiseGate ? 'Noise filter on' : 'Noise filter off' },
    { icon: ScanLine, title: 'AI scan', sub: active ? `${Math.round(active.progress * 100)}% of current sample` : `${finished} scanned` },
    { icon: ShieldCheck, title: 'Result & review', sub: `${finished} of ${items.length} ready` },
  ];

  const eta = avgTime && (queued || active)
    ? Math.max(1, Math.round(avgTime * (queued + (active ? 1 - active.progress : 0))))
    : null;

  return (
    <div className="pipeline">
      <ol className="steps">
        {steps.map((s, i) => {
          const state = i < step ? 'done' : i === step ? (running || allDone ? 'active' : 'paused') : 'todo';
          const Icon = state === 'done' ? Check : s.icon;
          return (
            <li key={s.title} className={`step ${state}`}>
              <span className="step-icon"><Icon size={15} strokeWidth={2.2} /></span>
              <span className="step-text">
                <strong>{s.title}</strong>
                <span>{s.sub}</span>
              </span>
            </li>
          );
        })}
      </ol>

      {active ? (
        <div className="now">
          <div className="now-thumb">
            <img src={active.url} alt="" />
            <span className="scanline" aria-hidden="true" />
          </div>
          <div className="now-text">
            <span className="now-label">Scanning</span>
            <strong title={active.file.name}>{active.file.name}</strong>
            <span className="progress"><span style={{ width: `${Math.max(3, active.progress * 100)}%` }} /></span>
            <span className="muted small">
              {Math.round(active.progress * 100)}%{eta ? ` · about ${eta < 60 ? `${eta} s` : `${Math.round(eta / 60)} min`} left in batch` : ''}
            </span>
          </div>
        </div>
      ) : (
        <div className="now idle">
          <span className="muted small">
            {queued > 0 ? (running ? 'Starting next sample…' : `Paused · ${queued} waiting`) : 'All samples processed'}
          </span>
        </div>
      )}
    </div>
  );
}
