import React, { useMemo, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import {
  AlertTriangle, Award, ChartSpline, CheckCircle2, Crosshair, Download, FileCheck2, FileText, Loader2, Play, RefreshCw, RotateCcw, SlidersVertical, Table2, Trash2,
} from 'lucide-react';
import { runThresholdAnalysis } from '../lib/api';
import { pairLabels, readLabelFiles } from '../lib/groundTruth';
import {
  F1_TIE, SWEEP_DEFAULTS, STEP_OPTIONS, buildThresholds, justify, pickBest, thresholdDigits,
} from '../lib/sweep';
import { exportThresholdCsv } from '../lib/export';
import MetricsChart from './MetricsChart';

const pct = (v) => (v == null ? '—' : `${(v * 100).toFixed(1)}%`);

export const initialThresholdState = {
  labels: {},             // stem → { name, text }
  emptyIfMissing: false,
  ...SWEEP_DEFAULTS,
  analysis: null,         // server response
  runKey: null,
};

function NumberField({ label, value, onChange, min, max, step }) {
  // keep what the user typed (e.g. "0.9" on the way to "0.95"); follow outside changes like Reset
  const [draft, setDraft] = useState(String(value));
  const [prev, setPrev] = useState(value);
  if (!Object.is(prev, value)) {
    setPrev(value);
    if (Number.isFinite(value) && parseFloat(draft) !== value) setDraft(String(value));
  }
  return (
    <label className="num-field">
      <span>{label}</span>
      <input
        type="number" inputMode="decimal" min={min} max={max} step={step} value={draft}
        onChange={(e) => { setDraft(e.target.value); const v = parseFloat(e.target.value); onChange(Number.isFinite(v) ? v : NaN); }}
      />
    </label>
  );
}

export default function ThresholdAnalysis({
  items, model, state, setState, onReanalyze, appliedThreshold, onApplyThreshold, staleCount,
}) {
  const [running, setRunning] = useState(false);
  const [error, setError] = useState(null);
  const set = (patch) => setState((s) => ({ ...s, ...patch }));
  const minThr = model?.limits.min_threshold ?? 0.5;

  const done = items.filter((it) => it.status === 'done' && it.result?.request_id);
  const pairing = useMemo(
    () => pairLabels(done, state.labels, state.emptyIfMissing),
    [done, state.labels, state.emptyIfMissing],
  );
  const objects = pairing.images.reduce((a, im) => a + im.boxes.length, 0);
  const labelCount = Object.keys(state.labels).length;

  const range = buildThresholds(state.start, state.end, state.step, minThr);
  const runKey = JSON.stringify([range.values, pairing.images.map((im) => [im.item.result.request_id, im.boxes.length])]);
  const analysis = state.analysis;
  const stale = analysis && state.runKey !== runKey;

  const { getRootProps, getInputProps, isDragActive, open } = useDropzone({
    onDrop: async (files) => {
      const read = await readLabelFiles(files);
      setState((s) => ({ ...s, labels: { ...s.labels, ...read } }));
    },
    accept: { 'text/plain': ['.txt'] },
    multiple: true,
    noClick: true,
    noDragEventsBubbling: true,   // keep the page-wide image drop overlay out of this
  });

  const run = async () => {
    setRunning(true);
    setError(null);
    try {
      const res = await runThresholdAnalysis(range.values, pairing.images.map((im) => ({
        request_id: im.item.result.request_id, ground_truth: im.boxes,
      })));
      set({ analysis: res, runKey });
    } catch (e) {
      setError(e.message);
    } finally {
      setRunning(false);
    }
  };

  const rows = analysis?.results ?? [];
  const digits = thresholdDigits(rows.map((r) => r.threshold));
  const best = rows.length ? pickBest(rows, analysis.objects) : null;
  const bestRow = best ? rows[best.index] : null;
  const missingNames = analysis?.missing?.length
    ? done.filter((it) => analysis.missing.includes(it.result.request_id)).map((it) => it.file.name)
    : [];
  const canRun = !running && range.values && pairing.images.length > 0;

  return (
    <div className="page threshold-page">
      <div className="page-head">
        <div>
          <h1>Threshold analysis</h1>
          <p className="muted">How detection quality changes with the confidence threshold, measured against your ground-truth labels.</p>
        </div>
        {analysis && (
          <div className="head-actions">
            <button className="btn" onClick={() => exportThresholdCsv(analysis, best?.index)}>
              <Download size={15} /> Export CSV
            </button>
          </div>
        )}
      </div>

      <div className="grid">
        {/* ── setup ─────────────────────────────────────────────── */}
        <div className="card sweep-setup">
          <div className="card-head"><h2><SlidersVertical size={17} /> Setup</h2></div>

          <section className="setup-section">
            <span className="section-label">1 · Ground-truth labels</span>
            <div {...getRootProps({ className: `label-drop ${isDragActive ? 'active' : ''}` })}>
              <input {...getInputProps()} aria-label="Add ground-truth label files" />
              <FileText size={20} />
              <span><strong>Drop label files</strong> or <button type="button" className="link-btn inline-link" onClick={open}>browse</button></span>
              <small>One .txt per image, same file name (e.g. <code>12.jpg</code> ↔ <code>12.txt</code>)</small>
            </div>
            <div className="label-status">
              <span className={`status-line ${pairing.images.length ? 'ok' : ''}`}>
                <FileCheck2 size={15} />
                <span><strong>{pairing.images.length}</strong> of {done.length} analysed sample{done.length === 1 ? '' : 's'} labelled · <strong>{objects.toLocaleString()}</strong> labelled object{objects === 1 ? '' : 's'}</span>
              </span>
              {labelCount > 0 && (
                <button className="icon-btn sm ghost" onClick={() => set({ labels: {} })} title="Remove all label files" aria-label="Remove all label files">
                  <Trash2 size={15} />
                </button>
              )}
            </div>
            {pairing.unused.length > 0 && (
              <p className="fine-print">{pairing.unused.length} label file{pairing.unused.length > 1 ? 's match' : ' matches'} no analysed sample ({pairing.unused.slice(0, 3).join(', ')}{pairing.unused.length > 3 ? ', …' : ''}).</p>
            )}
            {pairing.errors.length > 0 && (
              <p className="fine-print error-text">Could not read {pairing.errors.map((e) => `${e.name} (${e.error})`).join('; ')}.</p>
            )}
            <label className="switch">
              <input type="checkbox" checked={state.emptyIfMissing} onChange={(e) => set({ emptyIfMissing: e.target.checked })} />
              <span className="track" aria-hidden="true" />
              <span className="option-text">
                <span>Samples without labels are empty</span>
                <small>Count {pairing.unlabelled.length || 'unlabelled'} sample{pairing.unlabelled.length === 1 ? '' : 's'} without a label file as containing no bacteria, instead of leaving them out.</small>
              </span>
            </label>
          </section>

          <section className="setup-section">
            <span className="section-label">2 · Threshold range</span>
            <div className="range-fields">
              <NumberField label="From" value={state.start} min={minThr} max={0.999} step={0.005} onChange={(v) => set({ start: v })} />
              <NumberField label="To" value={state.end} min={minThr} max={0.999} step={0.005} onChange={(v) => set({ end: v })} />
            </div>
            <div className="step-field">
              <span>Step</span>
              <div className="segmented" role="radiogroup" aria-label="Step size">
                {STEP_OPTIONS.map((s) => (
                  <button key={s} role="radio" aria-checked={state.step === s} className={state.step === s ? 'active' : ''} onClick={() => set({ step: s })}>
                    {s}
                  </button>
                ))}
              </div>
            </div>
            {range.error ? (
              <p className="fine-print error-text">{range.error}</p>
            ) : (() => {
              const d = thresholdDigits(range.values);
              const v = range.values;
              return (
                <div className="thr-preview" aria-label="Thresholds to test">
                  <span className="fine-print">{v.length} threshold{v.length > 1 ? 's' : ''}:</span>
                  {v.length <= 12
                    ? v.map((t) => <span key={t} className="thr-chip">{t.toFixed(d)}</span>)
                    : <span className="thr-chip">{v[0].toFixed(d)} → {v[v.length - 1].toFixed(d)}</span>}
                </div>
              );
            })()}
            {(state.start !== SWEEP_DEFAULTS.start || state.end !== SWEEP_DEFAULTS.end || state.step !== SWEEP_DEFAULTS.step) && (
              <button className="link-btn reset-link" onClick={() => set(SWEEP_DEFAULTS)}><RotateCcw size={13} /> Reset to 0.95 – 0.98, step 0.01</button>
            )}
          </section>

          <div className="setup-run">
            <button className="btn primary run-btn" onClick={run} disabled={!canRun}>
              {running ? <Loader2 size={16} className="spin" /> : analysis ? <RefreshCw size={16} /> : <Play size={16} />}
              {running ? 'Analysing…' : analysis ? 'Run again' : 'Run analysis'}
            </button>
            <p className="fine-print">Re-scores the existing results at each threshold — no images are re-processed. One threshold applies to every detection.</p>
          </div>
        </div>

        {/* ── results ───────────────────────────────────────────── */}
        <div className="sweep-results">
          {error && <div className="alert error"><AlertTriangle size={18} /><span>{error}</span></div>}
          {missingNames.length > 0 && (
            <div className="alert warning">
              <AlertTriangle size={18} />
              <span>{missingNames.length} sample{missingNames.length > 1 ? 's were' : ' was'} left out because the server no longer holds {missingNames.length > 1 ? 'their' : 'its'} analysis (it was restarted). Re-analyse to include {missingNames.length > 1 ? 'them' : 'it'}.</span>
              <button className="btn small" onClick={onReanalyze}><RefreshCw size={14} /> Re-analyse</button>
            </div>
          )}
          {stale && !running && (
            <div className="alert info-alert">
              <RefreshCw size={16} />
              <span>Labels or range changed since this analysis. Run it again to update the results.</span>
            </div>
          )}

          {!analysis ? (
            <div className="card sweep-empty">
              <span className="hint-icon lg"><ChartSpline size={26} /></span>
              <h2>No analysis yet</h2>
              <p className="muted">
                {done.length === 0
                  ? 'Analyse some samples first, then add their ground-truth label files.'
                  : pairing.images.length === 0
                    ? 'Add ground-truth label files for your analysed samples, then run the analysis.'
                    : `Ready: ${pairing.images.length} labelled sample${pairing.images.length > 1 ? 's' : ''}, ${range.values?.length ?? 0} thresholds.`}
              </p>
              <ol className="how">
                <li><strong>Precision</strong> — share of detections that land on a labelled object.</li>
                <li><strong>Recall</strong> — share of labelled objects that are detected.</li>
                <li><strong>F1-score</strong> — balance of the two; the best threshold maximises it.</li>
              </ol>
            </div>
          ) : (
            <>
              {bestRow && (
                <div className={`card best-card ${stale ? 'is-stale' : ''}`}>
                  <div className="best-top">
                    <div className="best-id">
                      <span className="best-badge"><Award size={14} /> Recommended threshold</span>
                      <div className="best-value">{bestRow.threshold.toFixed(digits)}</div>
                    </div>
                    <dl className="best-metrics">
                      <div><dt>F1-score</dt><dd>{pct(bestRow.f1)}</dd></div>
                      <div><dt>Precision</dt><dd>{pct(bestRow.precision)}</dd></div>
                      <div><dt>Recall</dt><dd>{pct(bestRow.recall)}</dd></div>
                      <div><dt>Detections</dt><dd>{bestRow.detections.toLocaleString()}</dd></div>
                    </dl>
                  </div>
                  <div className="best-apply">
                    {appliedThreshold != null && Math.abs(appliedThreshold - bestRow.threshold) < 1e-9 ? (
                      <>
                        <span className="applied-note"><CheckCircle2 size={16} /> In use for analysis</span>
                        {staleCount > 0 && (
                          <button className="btn small" onClick={onReanalyze}>
                            <RefreshCw size={14} /> Re-analyse {staleCount} sample{staleCount > 1 ? 's' : ''}
                          </button>
                        )}
                        <span className="fine-print">Boxes found at this threshold are then screened for noise and checked with SAM.</span>
                      </>
                    ) : (
                      <>
                        <button className="btn primary small" onClick={() => onApplyThreshold(bestRow.threshold)} disabled={stale}>
                          <Crosshair size={14} /> Use {bestRow.threshold.toFixed(digits)} for analysis
                        </button>
                        <span className="fine-print">
                          {appliedThreshold != null
                            ? `Currently ${appliedThreshold}. `
                            : 'Currently the model\'s default thresholds. '}
                          Detection runs at this threshold, then noise screening and SAM verification.
                        </span>
                      </>
                    )}
                  </div>
                  <div className="best-why">
                    <span className="section-label">Why this threshold</span>
                    <p>{justify(rows, best, analysis.objects, digits)}</p>
                    <p className="fine-print">
                      Criterion: highest F1-score. When F1-scores are within {+(F1_TIE * 100).toFixed(2)} points, higher precision and recall combined, then higher precision, then the detection count closest to the {analysis.objects.toLocaleString()} labelled objects decide.
                      Based on {analysis.images} sample{analysis.images > 1 ? 's' : ''}.
                    </p>
                  </div>
                </div>
              )}
              {!bestRow && (
                <div className="alert warning"><AlertTriangle size={18} /><span>No labelled objects in the selected samples, so recall and F1-score cannot be computed. Add label files that contain objects.</span></div>
              )}

              <div className="card">
                <div className="card-head"><h2><ChartSpline size={17} /> Precision, recall and F1-score by threshold</h2></div>
                <MetricsChart rows={rows} bestIndex={best?.index} digits={digits} />
              </div>

              <div className="card table-card">
                <div className="card-head"><h2><Table2 size={17} /> Comparison</h2></div>
                <div className="table-wrap scroll-x">
                  <table className="table sweep-table">
                    <thead>
                      <tr>
                        <th>Threshold</th>
                        <th className="num">Precision</th>
                        <th className="num">Recall</th>
                        <th className="num">F1-score</th>
                        <th className="num">Detection count</th>
                      </tr>
                    </thead>
                    <tbody>
                      {rows.map((r, i) => {
                        const isBest = i === best?.index;
                        return (
                          <tr key={r.threshold} className={isBest ? 'best-row' : ''} aria-current={isBest ? 'true' : undefined}>
                            <td>
                              <span className="thr-cell">
                                <strong className="tabular">{r.threshold.toFixed(digits)}</strong>
                                {isBest && <span className="best-pill" aria-label="Best threshold"><Award size={12} /><span className="hide-sm">Best</span></span>}
                              </span>
                            </td>
                            <td className="num">{pct(r.precision)}</td>
                            <td className="num">{pct(r.recall)}</td>
                            <td className="num strong">{pct(r.f1)}</td>
                            <td className="num">{r.detections.toLocaleString()}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
                <p className="fine-print table-note">
                  A detection is correct when it lands on a labelled object (either box's centre inside the other). {analysis.objects.toLocaleString()} labelled objects in {analysis.images} sample{analysis.images > 1 ? 's' : ''}.
                </p>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
