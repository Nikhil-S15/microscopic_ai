import React, { useEffect, useState } from 'react';
import { Cpu, Filter, Gauge, Layers, RefreshCw, ScanSearch, ShieldCheck, X } from 'lucide-react';

const SENSITIVITY = [
  { value: -0.05, label: 'Strict', hint: 'Only high-certainty detections. Fewer false positives.' },
  { value: 0, label: 'Balanced', hint: 'Recommended. Default detection thresholds.' },
  { value: 0.1, label: 'Sensitive', hint: 'Catches faint or partial organisms. Expect more items to review.' },
];

/** Number input that tolerates half-typed values and reports only valid numbers. */
function NumberInput({ value, onChange, min, max, step, suffix, label, disabled }) {
  const [draft, setDraft] = useState(String(value));
  const [prev, setPrev] = useState(value);
  if (!Object.is(prev, value)) {
    setPrev(value);
    if (parseFloat(draft) !== value) setDraft(String(value));
  }
  return (
    <label className="setting-num">
      <span>{label}</span>
      <span className="setting-num-input">
        <input
          type="number" inputMode="decimal" min={min} max={max} step={step} value={draft} disabled={disabled}
          onChange={(e) => {
            setDraft(e.target.value);
            const v = parseFloat(e.target.value);
            if (Number.isFinite(v) && v >= min && v <= max) onChange(v);
          }}
        />
        {suffix && <small>{suffix}</small>}
      </span>
    </label>
  );
}

export default function SettingsDrawer({ open, onClose, model, settings, onChange, staleCount, onReanalyze }) {
  const set = (patch) => onChange({ ...settings, ...patch });
  const sam = model.screening?.sam;
  const minThr = model.limits.min_threshold ?? 0.5;

  useEffect(() => {
    if (!open) return undefined;
    const onKey = (e) => e.key === 'Escape' && onClose();
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  const fixed = settings.threshold != null;

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
            <h3><Gauge size={15} /> 1 · Detection threshold</h3>
            <label className="switch">
              <input type="checkbox" checked={fixed} onChange={(e) => set({ threshold: e.target.checked ? 0.96 : null })} />
              <span className="track" aria-hidden="true" />
              <span className="option-text">
                <span>Fixed confidence threshold</span>
                <small>One threshold for every detection — pick it in Threshold analysis. Off: the model's default thresholds with the sensitivity below.</small>
              </span>
            </label>
            {fixed ? (
              <NumberInput label="Threshold" value={settings.threshold} min={minThr} max={0.999} step={0.005}
                           onChange={(v) => set({ threshold: v })} />
            ) : (
              <>
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
              </>
            )}
            <label className="switch">
              <input type="checkbox" checked={settings.noiseGate} disabled={!model.noise_gate}
                     onChange={(e) => set({ noiseGate: e.target.checked })} />
              <span className="track" aria-hidden="true" />
              <span className="option-text">
                <span>Skip empty background</span>
                <small>Pre-filters empty background and debris windows before the AI scan. Faster.</small>
              </span>
            </label>
          </section>

          <section>
            <h3><Filter size={15} /> 2 · Noise screening</h3>
            <label className="switch">
              <input type="checkbox" checked={settings.boxScreening} disabled={!model.screening?.box_screening}
                     onChange={(e) => set({ boxScreening: e.target.checked })} />
              <span className="track" aria-hidden="true" />
              <span className="option-text">
                <span>Screen every box for noise</span>
                <small>Each detected box is classified as noise or valid by the noise model; noise boxes are removed.</small>
              </span>
            </label>
            <NumberInput label="Remove when noise probability is above" value={settings.noiseCutoff} min={0} max={1} step={0.05}
                         disabled={!settings.boxScreening} onChange={(v) => set({ noiseCutoff: v })} />
          </section>

          <section>
            <h3><Layers size={15} /> 3 · Mask verification</h3>
            <label className="switch">
              <input type="checkbox" checked={settings.samVerify} disabled={!sam?.available}
                     onChange={(e) => set({ samVerify: e.target.checked })} />
              <span className="track" aria-hidden="true" />
              <span className="option-text">
                <span>Verify boxes with SAM</span>
                <small>
                  {sam?.available
                    ? 'Each remaining box prompts the segmentation model; a box is kept only if an object mask fills enough of it. Masks are saved.'
                    : 'Unavailable — the segmentation weights are not installed on the server (backend/scripts/download_sam.py).'}
                </small>
              </span>
            </label>
            <div className="setting-pair">
              <NumberInput label="Mask threshold (logit)" value={settings.maskThreshold} min={-20} max={20} step={0.5}
                           disabled={!settings.samVerify} onChange={(v) => set({ maskThreshold: v })} />
              <NumberInput label="Min. mask coverage" value={Math.round(settings.minCoverage * 100)} min={0} max={100} step={5} suffix="%"
                           disabled={!settings.samVerify} onChange={(v) => set({ minCoverage: v / 100 })} />
            </div>
            <p className="fine-print">Mask pixels are those with a logit above the threshold (0 = SAM's default). Coverage = mask area ÷ box area.</p>
          </section>

          <section>
            <h3><ShieldCheck size={15} /> Models</h3>
            <dl className="facts">
              <dt><ScanSearch size={13} className="inline" /> Detector</dt><dd>{model.architecture}</dd>
              <dt>Segmentation</dt><dd>{sam?.available ? sam.model : 'Not installed'}</dd>
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
