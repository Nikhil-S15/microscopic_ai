import React, { useEffect } from 'react';
import { Cpu, Filter, Gauge, RefreshCw, ShieldCheck, X } from 'lucide-react';

const SENSITIVITY = [
  { value: -0.05, label: 'Strict', hint: 'Only high-certainty detections. Fewer false positives.' },
  { value: 0, label: 'Balanced', hint: 'Recommended. Default detection thresholds.' },
  { value: 0.1, label: 'Sensitive', hint: 'Catches faint or partial organisms. Expect more items to review.' },
];

export default function SettingsDrawer({ open, onClose, model, settings, onChange, staleCount, onReanalyze }) {
  const set = (patch) => onChange({ ...settings, ...patch });

  useEffect(() => {
    if (!open) return undefined;
    const onKey = (e) => e.key === 'Escape' && onClose();
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  return (
    <>
      <div className={`scrim ${open ? 'open' : ''}`} onClick={onClose} aria-hidden="true" />
      <aside className={`drawer ${open ? 'open' : ''}`} aria-label="Analysis settings" aria-hidden={!open} inert={open ? undefined : ''}>
        <div className="drawer-head">
          <div>
            <h2>Analysis settings</h2>
            <p className="muted">Applied to samples analysed from now on</p>
          </div>
          <button className="icon-btn ghost" onClick={onClose} aria-label="Close settings"><X size={18} /></button>
        </div>

        <div className="drawer-body">
          <section>
            <h3><Gauge size={15} /> Detection sensitivity</h3>
            <div className="segmented" role="radiogroup" aria-label="Detection sensitivity">
              {SENSITIVITY.map((o) => (
                <button
                  key={o.label}
                  role="radio"
                  aria-checked={settings.sensitivity === o.value}
                  className={settings.sensitivity === o.value ? 'active' : ''}
                  onClick={() => set({ sensitivity: o.value })}
                >
                  {o.label}
                </button>
              ))}
            </div>
            <p className="fine-print">{SENSITIVITY.find((o) => o.value === settings.sensitivity)?.hint}</p>
          </section>

          <section>
            <h3><Filter size={15} /> Pre-processing</h3>
            <label className="switch">
              <input type="checkbox" checked={settings.noiseGate} disabled={!model.noise_gate}
                     onChange={(e) => set({ noiseGate: e.target.checked })} />
              <span className="track" aria-hidden="true" />
              <span className="option-text">
                <span>Noise &amp; debris filter</span>
                <small>Skips empty background and debris before AI analysis. Faster, with fewer false positives.</small>
              </span>
            </label>
          </section>

          <section>
            <h3><ShieldCheck size={15} /> Model</h3>
            <dl className="facts">
              <dt>Architecture</dt><dd>{model.architecture}</dd>
              <dt><Cpu size={13} className="inline" /> Compute</dt><dd>{model.device.toUpperCase()}</dd>
            </dl>
          </section>
        </div>

        <div className="drawer-foot">
          {staleCount > 0 ? (
            <>
              <span className="fine-print">{staleCount} sample{staleCount > 1 ? 's were' : ' was'} analysed with earlier settings.</span>
              <button className="btn primary" onClick={onReanalyze}><RefreshCw size={15} /> Re-analyse</button>
            </>
          ) : (
            <span className="fine-print">All samples use the current settings.</span>
          )}
        </div>
      </aside>
    </>
  );
}
