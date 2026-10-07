import React from 'react';
import { Check, Filter, ImageUp, Layers, ScanLine, ShieldCheck } from 'lucide-react';

/**
 * Detection process: where the batch is in the pipeline (intake → scan →
 * noise screening → mask verification → review), plus live progress of the
 * sample currently being scanned.
 */
export default function Pipeline({ items, running, settings, samAvailable, avgTime }) {
  const queued = items.filter((i) => i.status === 'queued').length;
  const active = items.find((i) => i.status === 'analyzing');
  const finished = items.filter((i) => i.status === 'done' || i.status === 'error').length;
  const allDone = !active && queued === 0;
  const sam = settings.samVerify && samAvailable;

  const sc = items.filter((i) => i.status === 'done' && i.result.screening).map((i) => i.result.screening);
  const removedNoise = sc.reduce((a, s) => a + s.removed_noise, 0);
  const removedMask = sc.reduce((a, s) => a + s.removed_mask, 0);
  const detected = sc.reduce((a, s) => a + s.boxes_detected, 0);

  // step in progress (0-based); 5 = complete. The server reports detection as 0–85% when SAM runs.
  const scanEnd = sam ? 0.85 : 0.99;
  const step = allDone ? 5 : active ? (active.progress <= 0 ? 1 : active.progress < scanEnd ? 1 : sam ? 3 : 2) : 0;
  const steps = [
    { icon: ImageUp, title: 'Sample intake', sub: `${items.length} received` },
    {
      icon: ScanLine,
      title: 'AI scan',
      sub: active && step === 1 ? `${Math.round((active.progress / scanEnd) * 100)}% of current sample`
        : `${settings.threshold != null ? `Threshold ${settings.threshold}` : 'Default thresholds'} · ${detected} box${detected === 1 ? '' : 'es'} found`,
    },
    {
      icon: Filter,
      title: 'Noise screening',
      sub: settings.boxScreening ? `${removedNoise} removed as noise` : 'Off',
      off: !settings.boxScreening,
    },
    {
      icon: Layers,
      title: 'Mask verification',
      sub: sam ? `${removedMask} removed · masks saved` : samAvailable ? 'Off' : 'Not installed',
      off: !sam,
    },
    { icon: ShieldCheck, title: 'Result & review', sub: `${finished} of ${items.length} ready` },
  ];

  const eta = avgTime && (queued || active)
    ? Math.max(1, Math.round(avgTime * (queued + (active ? 1 - active.progress : 0))))
    : null;

  return (
    <div className="pipeline">
      <ol className="steps">
        {steps.map((s, i) => {
          const state = s.off && i < step ? 'skipped'
            : i < step ? 'done' : i === step ? (running || allDone ? 'active' : 'paused') : 'todo';
          const Icon = state === 'done' ? Check : s.icon;
          return (
            <li key={s.title} className={`step ${state} ${s.off ? 'off' : ''}`}>
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
            <span className="now-label">{step === 3 ? 'Verifying masks' : 'Scanning'}</span>
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
