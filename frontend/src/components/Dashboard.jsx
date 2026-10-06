import React, { useEffect, useRef, useState } from 'react';
import {
  Activity, ChevronDown, ChevronRight, Download, FileSpreadsheet, ImagePlus, Info, Pause, Play,
  RotateCw, ShieldAlert, ShieldCheck, Target, Trash2, X, Microscope, Loader2,
} from 'lucide-react';
import ResultBadge from './ResultBadge';
import ConfidenceRing from './ConfidenceRing';
import ConfidenceHistogram from './ConfidenceHistogram';
import Pipeline from './Pipeline';
import { num, pct, reviewProgress, sampleMetrics, seconds } from '../lib/format';
import { exportDetectionsCsv, exportSummaryCsv } from '../lib/export';

const SENS_LABEL = { '-0.05': 'Strict', 0: 'Balanced', 0.1: 'Sensitive' };

function ExportMenu({ disabled, items }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);
  useEffect(() => {
    if (!open) return undefined;
    const close = (e) => { if (!ref.current?.contains(e.target)) setOpen(false); };
    const esc = (e) => e.key === 'Escape' && setOpen(false);
    document.addEventListener('mousedown', close);
    document.addEventListener('keydown', esc);
    return () => { document.removeEventListener('mousedown', close); document.removeEventListener('keydown', esc); };
  }, [open]);
  return (
    <div className="menu-wrap" ref={ref}>
      <button className="btn" onClick={() => setOpen((o) => !o)} disabled={disabled} aria-haspopup="menu" aria-expanded={open}>
        <Download size={15} /> Export <ChevronDown size={14} />
      </button>
      {open && (
        <div className="menu" role="menu">
          <button role="menuitem" onClick={() => { exportSummaryCsv(items); setOpen(false); }}>
            <FileSpreadsheet size={16} />
            <span><strong>Sample report</strong><small>Result and metrics per sample · CSV</small></span>
          </button>
          <button role="menuitem" onClick={() => { exportDetectionsCsv(items); setOpen(false); }}>
            <Target size={16} />
            <span><strong>Detection list</strong><small>Every location and confidence · CSV</small></span>
          </button>
        </div>
      )}
    </div>
  );
}

function ReviewCell({ it }) {
  const p = reviewProgress(it);
  if (!p.total) return <span className="faint">—</span>;
  const done = p.total - p.pending;
  return (
    <span className="review-cell" title={`${p.confirmed} confirmed · ${p.rejected} rejected · ${p.pending} pending`}>
      <span className="mini-track"><span style={{ width: `${(done / p.total) * 100}%` }} /></span>
      <span className="faint">{done}/{p.total}</span>
    </span>
  );
}

export default function Dashboard({
  items, model, settings, running, staleCount, onAdd, onToggleRunning, onReanalyze, onOpen, onRemove, onRetry, onClear,
}) {
  const done = items.filter((it) => it.status === 'done');
  const pending = items.filter((it) => it.status === 'queued' || it.status === 'analyzing').length;
  const failed = items.filter((it) => it.status === 'error').length;
  const metrics = new Map(done.map((it) => [it.id, sampleMetrics(it, model?.high_confidence)]));
  const positives = done.filter((it) => metrics.get(it.id).detected).length;
  const clearCount = done.length - positives;
  const totalDet = done.reduce((a, it) => a + metrics.get(it.id).count, 0);
  const allConf = done.flatMap((it) => metrics.get(it.id).confidences);
  const meanConf = allConf.length ? allConf.reduce((a, b) => a + b, 0) / allConf.length : null;
  const highShare = allConf.length ? allConf.filter((c) => c >= (model?.high_confidence ?? 0.95)).length / allConf.length : null;
  const avgTime = done.length ? done.reduce((a, it) => a + it.result.processing_time, 0) / done.length : null;

  let hero;
  if (!done.length) {
    hero = { tone: 'pending', Icon: Loader2, title: 'Analysis in progress', sub: 'The batch result appears as soon as the first sample is scanned.' };
  } else if (positives > 0) {
    hero = { tone: 'detected', Icon: ShieldAlert, title: 'Bacteria detected', sub: `Present in ${positives} of ${done.length} analysed sample${done.length === 1 ? '' : 's'} · ${num(totalDet)} detection${totalDet === 1 ? '' : 's'} in total` };
  } else {
    hero = { tone: 'clear', Icon: ShieldCheck, title: 'No bacteria detected', sub: `All ${done.length} analysed sample${done.length === 1 ? ' is' : 's are'} clear` };
  }

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>Analysis dashboard</h1>
          <p className="muted">
            {items.length} sample{items.length === 1 ? '' : 's'}
            {pending > 0 && <> · {running ? `${pending} in progress` : `paused, ${pending} waiting`}</>}
            {failed > 0 && <> · <span className="error-text">{failed} failed</span></>}
          </p>
        </div>
        <div className="head-actions">
          {pending > 0 && (
            <button className="btn" onClick={onToggleRunning}>
              {running ? <><Pause size={15} /> Pause</> : <><Play size={15} /> Resume</>}
            </button>
          )}
          <ExportMenu disabled={!done.length} items={items} />
          <button className="icon-btn" onClick={onClear} title="Clear all samples" aria-label="Clear all samples"><Trash2 size={16} /></button>
        </div>
      </div>

      {staleCount > 0 && (
        <div className="alert warning">
          <Info size={17} />
          <span>Settings changed — {staleCount} sample{staleCount > 1 ? 's were' : ' was'} analysed with the previous settings.</span>
          <button className="btn small" onClick={onReanalyze}><RotateCw size={13} /> Re-analyse</button>
        </div>
      )}

      <div className="grid">
        {/* ── RESULT (main focus) ─────────────────────────────── */}
        <section className={`card result-hero ${hero.tone}`} aria-live="polite">
          <div className="hero-main">
            <span className="hero-icon"><hero.Icon size={30} strokeWidth={1.9} className={hero.tone === 'pending' ? 'spin' : ''} /></span>
            <div>
              <span className="section-label">Batch result</span>
              <h2 className="hero-title">{hero.title}</h2>
              <p className="hero-sub">{hero.sub}</p>
            </div>
          </div>
          <div className="breakdown">
            <div className="bd-bar" role="img" aria-label={`${positives} positive, ${clearCount} clear, ${pending} pending, ${failed} failed`}>
              {[['detected', positives], ['clear', clearCount], ['pending', pending], ['failed', failed]].filter(([, n]) => n > 0).map(([k, n]) => (
                <span key={k} className={`bd-seg ${k}`} style={{ flexGrow: n }} />
              ))}
            </div>
            <ul className="bd-legend">
              <li><span className="bd-dot detected" /> Bacteria detected <strong>{positives}</strong></li>
              <li><span className="bd-dot clear" /> No bacteria <strong>{clearCount}</strong></li>
              {pending > 0 && <li><span className="bd-dot pending" /> In progress <strong>{pending}</strong></li>}
              {failed > 0 && <li><span className="bd-dot failed" /> Failed <strong>{failed}</strong></li>}
            </ul>
          </div>
          <dl className="hero-stats">
            <div><dt>Samples analysed</dt><dd>{done.length}<small>/{items.length}</small></dd></div>
            <div><dt>Positive samples</dt><dd>{positives}</dd></div>
            <div><dt>Total detections</dt><dd>{num(totalDet)}</dd></div>
            <div><dt>Avg. time / sample</dt><dd>{avgTime == null ? '—' : avgTime.toFixed(1)}<small>{avgTime == null ? '' : ' s'}</small></dd></div>
          </dl>
        </section>

        {/* ── CONFIDENCE ──────────────────────────────────────── */}
        <section className="card confidence-card">
          <div className="card-head">
            <h2><Activity size={16} /> Detection confidence</h2>
          </div>
          <div className="conf-body">
            <ConfidenceRing value={meanConf} size={124} label="Average" />
            <dl className="conf-facts">
              <div><dt>High-confidence (≥{Math.round((model?.high_confidence ?? 0.95) * 100)}%)</dt><dd>{pct(highShare)}</dd></div>
              <div><dt>Lowest</dt><dd>{allConf.length ? pct(Math.min(...allConf)) : '—'}</dd></div>
              <div><dt>Detections scored</dt><dd>{num(allConf.length)}</dd></div>
            </dl>
          </div>
          <ConfidenceHistogram confidences={allConf} />
        </section>

        {/* ── PROCESS + SUPPORTING INFO ───────────────────────── */}
        <div className="col-side">
          <section className="card">
            <div className="card-head"><h2><Microscope size={16} /> Detection process</h2></div>
            <Pipeline items={items} running={running} noiseGate={settings.noiseGate} avgTime={avgTime} />
          </section>

          <section className="card info-card">
            <div className="card-head"><h2><Info size={16} /> Supporting information</h2></div>
            <dl className="facts">
              <dt>Model</dt><dd>{model?.architecture ?? '—'}</dd>
              <dt>Sensitivity</dt><dd>{SENS_LABEL[String(settings.sensitivity)] ?? 'Custom'}</dd>
              <dt>Compute</dt><dd>{model?.device.toUpperCase() ?? '—'}</dd>
            </dl>
            <p className="fine-print">
              AI results support, and do not replace, expert review. Confirm or reject each detection
              in Review before reporting.
            </p>
          </section>
        </div>

        {/* ── SAMPLES + UPLOAD ────────────────────────────────── */}
        <section className="card samples-card">
          <div className="card-head">
            <h2>Samples</h2>
            <button className="btn small" onClick={onAdd}><ImagePlus size={14} /> Upload more</button>
          </div>
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th>Sample</th>
                  <th>Result</th>
                  <th className="num">Detections</th>
                  <th className="num hide-md">Confidence</th>
                  <th className="hide-sm">Reviewed</th>
                  <th aria-label="Actions" />
                </tr>
              </thead>
              <tbody>
                {items.map((it) => {
                  const m = metrics.get(it.id);
                  const isDone = it.status === 'done';
                  return (
                    <tr
                      key={it.id}
                      className={isDone ? 'clickable' : ''}
                      onClick={() => isDone && onOpen(it.id)}
                      onKeyDown={(e) => isDone && e.key === 'Enter' && onOpen(it.id)}
                      tabIndex={isDone ? 0 : undefined}
                    >
                      <td>
                        <span className="sample">
                          <span className="thumb">
                            <img src={it.url} alt="" loading="lazy" onError={(e) => { e.currentTarget.style.visibility = 'hidden'; }} />
                            {it.status === 'analyzing' && <span className="scanline" aria-hidden="true" />}
                          </span>
                          <span className="sample-text">
                            <span className="sample-name" title={it.file.name}>{it.file.name}</span>
                            <span className="faint small">
                              {isDone ? `${it.result.image.width}×${it.result.image.height} · ${seconds(it.result.processing_time)}`
                                : it.status === 'analyzing' ? `Scanning ${Math.round(it.progress * 100)}%`
                                  : it.status === 'error' ? it.error : 'Waiting in queue'}
                            </span>
                          </span>
                        </span>
                      </td>
                      <td><ResultBadge item={it} metrics={m} /></td>
                      <td className="num strong">{isDone ? m.count : ''}</td>
                      <td className="num hide-md">{isDone ? pct(m.meanConfidence) : ''}</td>
                      <td className="hide-sm">{isDone ? <ReviewCell it={it} /> : ''}</td>
                      <td className="row-actions" onClick={(e) => e.stopPropagation()}>
                        {it.status === 'error' && (
                          <button className="icon-btn ghost sm" onClick={() => onRetry(it.id)} title="Retry" aria-label={`Retry ${it.file.name}`}><RotateCw size={15} /></button>
                        )}
                        <button className="icon-btn ghost sm reveal" onClick={() => onRemove(it.id)} title="Remove" aria-label={`Remove ${it.file.name}`}><X size={15} /></button>
                        {isDone && <ChevronRight size={17} className="chev" aria-hidden="true" />}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>
      </div>
    </div>
  );
}
