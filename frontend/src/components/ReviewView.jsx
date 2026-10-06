import React, { useEffect, useRef, useState } from 'react';
import {
  ArrowLeft, Check, ChevronLeft, ChevronRight, ImageDown, Keyboard, RotateCcw, ShieldAlert, ShieldCheck,
} from 'lucide-react';
import ImageViewer from './ImageViewer';
import ConcentrationGrid from './ConcentrationGrid';
import Inspector from './Inspector';
import ConfidenceRing from './ConfidenceRing';
import { exportAnnotatedPng } from '../lib/export';
import { num, pct, reviewProgress, sampleMetrics, seconds } from '../lib/format';

const FILTERS = [
  { id: 'all', label: 'All' },
  { id: 'pending', label: 'Pending' },
  { id: 'confirmed', label: 'Confirmed' },
  { id: 'rejected', label: 'Rejected' },
];

function readTab() {
  try { return sessionStorage.getItem('review-tab') || 'fov'; } catch { return 'fov'; }
}

export default function ReviewView({ item, items, model, onSelect, onSetReview, onBack }) {
  const { result, review } = item;
  const [tab, setTab] = useState(readTab);
  const [filter, setFilter] = useState('all');
  const [showRejected, setShowRejected] = useState(true);
  const [selectedId, setSelectedId] = useState(null);
  const [exporting, setExporting] = useState(false);
  const [showKeys, setShowKeys] = useState(false);
  const viewerRef = useRef(null);
  const gridRef = useRef(null);

  useEffect(() => { try { sessionStorage.setItem('review-tab', tab); } catch { /* ignore */ } }, [tab]);

  const idx = items.findIndex((it) => it.id === item.id);
  const prev = items[idx - 1];
  const next = items[idx + 1];
  const m = sampleMetrics(item, model?.high_confidence);
  const progress = reviewProgress(item);

  const stateOf = (d) => review[d.id] ?? 'pending';
  const all = result.detections;
  const filterCounts = Object.fromEntries(FILTERS.map((f) => [f.id, f.id === 'all' ? all.length : all.filter((d) => stateOf(d) === f.id).length]));
  const listed = tab === 'grid'
    ? (filter === 'all' ? all : all.filter((d) => stateOf(d) === filter))
    : (showRejected ? all : all.filter((d) => stateOf(d) !== 'rejected'));
  const selected = all.find((d) => d.id === selectedId) ?? null;

  const select = (id, focus = tab === 'fov') => {
    setSelectedId(id);
    if (id == null) return;
    const det = all.find((d) => d.id === id);
    if (focus && det) viewerRef.current?.focusBox(det.bbox);
  };

  /** Apply a decision, then move to the next pending detection in the current list. */
  const decide = (id, state) => {
    onSetReview(item.id, [id], state);
    if (state === 'pending') return;
    const pos = listed.findIndex((d) => d.id === id);
    const after = [...listed.slice(pos + 1), ...listed.slice(0, pos)]
      .find((d) => d.id !== id && (review[d.id] ?? 'pending') === 'pending');
    if (after) select(after.id);
  };

  const step = (delta) => {
    if (!listed.length) return;
    const pos = listed.findIndex((d) => d.id === selectedId);
    const nextPos = pos === -1 ? (delta > 0 ? 0 : listed.length - 1) : Math.min(listed.length - 1, Math.max(0, pos + delta));
    select(listed[nextPos].id);
  };

  const gridColumns = () => {
    const el = gridRef.current;
    return el ? getComputedStyle(el).gridTemplateColumns.split(' ').length || 1 : 1;
  };

  useEffect(() => {
    const onKey = (e) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.target.closest('input, textarea, select, [role="menu"]') || document.querySelector('.drawer.open')) return;
      const k = e.key;
      if (k === 'ArrowRight' || k === 'j') { e.preventDefault(); step(1); }
      else if (k === 'ArrowLeft' || k === 'k') { e.preventDefault(); step(-1); }
      else if (tab === 'grid' && k === 'ArrowDown') { e.preventDefault(); step(gridColumns()); }
      else if (tab === 'grid' && k === 'ArrowUp') { e.preventDefault(); step(-gridColumns()); }
      else if ((k === 'c' || k === 'C') && selectedId != null) { e.preventDefault(); decide(selectedId, 'confirmed'); }
      else if ((k === 'r' || k === 'R' || k === 'x') && selectedId != null) { e.preventDefault(); decide(selectedId, 'rejected'); }
      else if ((k === 'u' || k === 'U') && selectedId != null) { e.preventDefault(); decide(selectedId, 'pending'); }
      else if (k === ']' && next) onSelect(next.id);
      else if (k === '[' && prev) onSelect(prev.id);
      else if (k === '1') setTab('fov');
      else if (k === '2') setTab('grid');
      else if (k === '?') setShowKeys((v) => !v);
      else if (k === 'Escape') { setSelectedId(null); viewerRef.current?.fit(); setShowKeys(false); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  });

  const pendingListed = listed.filter((d) => stateOf(d) === 'pending').map((d) => d.id);
  const ResultIcon = m.detected ? ShieldAlert : ShieldCheck;

  return (
    <div className="review">
      <div className="review-head">
        <button className="btn ghost back" onClick={onBack}><ArrowLeft size={16} /> <span className="hide-sm">Dashboard</span></button>
        <div className="review-title">
          <h1 title={item.file.name}>{item.file.name}</h1>
          <span className="muted">{result.image.width} × {result.image.height} px · scanned in {seconds(result.processing_time)}</span>
        </div>
        <div className="pager">
          <button className="icon-btn" onClick={() => prev && onSelect(prev.id)} disabled={!prev} aria-label="Previous sample" title="Previous sample ( [ )"><ChevronLeft size={16} /></button>
          <span className="muted tabular">{idx + 1} / {items.length}</span>
          <button className="icon-btn" onClick={() => next && onSelect(next.id)} disabled={!next} aria-label="Next sample" title="Next sample ( ] )"><ChevronRight size={16} /></button>
        </div>
        <button className="icon-btn" onClick={() => setShowKeys((v) => !v)} aria-label="Keyboard shortcuts" title="Keyboard shortcuts (?)" aria-expanded={showKeys}>
          <Keyboard size={16} />
        </button>
        <button
          className="btn"
          disabled={exporting}
          onClick={async () => { setExporting(true); try { await exportAnnotatedPng(item); } finally { setExporting(false); } }}
        >
          <ImageDown size={15} /> <span className="hide-sm">{exporting ? 'Exporting…' : 'Annotated image'}</span>
        </button>
      </div>

      {showKeys && (
        <div className="keys-panel" role="note">
          <span><kbd>←</kbd><kbd>→</kbd> previous / next detection</span>
          <span><kbd>C</kbd> confirm</span>
          <span><kbd>R</kbd> reject</span>
          <span><kbd>U</kbd> undo</span>
          <span><kbd>[</kbd><kbd>]</kbd> previous / next sample</span>
          <span><kbd>1</kbd><kbd>2</kbd> switch view</span>
          <span><kbd>Esc</kbd> fit / deselect</span>
        </div>
      )}

      {/* ── RESULT (main focus) ─────────────────────────────── */}
      <section className={`result-strip ${m.detected ? 'detected' : 'clear'}`} aria-live="polite">
        <div className="rs-main">
          <span className="hero-icon"><ResultIcon size={28} strokeWidth={1.9} /></span>
          <div>
            <span className="section-label">Sample result</span>
            <h2 className="hero-title">{m.detected ? 'Bacteria detected' : 'No bacteria detected'}</h2>
            <p className="hero-sub">
              {m.detected
                ? `${num(m.count)} detection${m.count === 1 ? '' : 's'} in this field of view${progress.rejected ? ` · ${progress.rejected} rejected in review` : ''}`
                : progress.rejected ? `All ${progress.rejected} detections were rejected in review` : 'The field of view appears clear'}
            </p>
          </div>
        </div>
        <dl className="rs-metrics">
          <div><dt>Detections</dt><dd>{num(m.count)}</dd></div>
          <div><dt>Peak confidence</dt><dd>{pct(m.maxConfidence)}</dd></div>
          <div><dt>Area coverage</dt><dd>{pct(m.coverage, 2)}</dd></div>
          <div><dt>Reviewed</dt><dd>{progress.total ? Math.round(((progress.total - progress.pending) / progress.total) * 100) : 100}<small>%</small></dd></div>
        </dl>
        <ConfidenceRing value={m.meanConfidence} size={96} stroke={8} label="Mean conf." />
      </section>

      <div className="review-tabs">
        <div className="seg-tabs" role="tablist">
          <button role="tab" aria-selected={tab === 'fov'} className={tab === 'fov' ? 'active' : ''} onClick={() => setTab('fov')}>
            Full FOV
          </button>
          <button role="tab" aria-selected={tab === 'grid'} className={tab === 'grid' ? 'active' : ''} onClick={() => setTab('grid')}>
            Digital concentration
            {progress.pending > 0 && <span className="count-badge">{progress.pending}</span>}
          </button>
        </div>
        {tab === 'grid' && (
          <div className="grid-tools">
            <div className="chips" role="radiogroup" aria-label="Filter by review state">
              {FILTERS.map((f) => (
                <button key={f.id} role="radio" aria-checked={filter === f.id} className={`chip ${filter === f.id ? 'on' : ''}`} onClick={() => setFilter(f.id)}>
                  {f.label} <span className="chip-count">{filterCounts[f.id]}</span>
                </button>
              ))}
            </div>
            <button className="btn small" disabled={!pendingListed.length} onClick={() => onSetReview(item.id, pendingListed, 'confirmed')}>
              <Check size={14} /> Confirm {pendingListed.length} pending
            </button>
            <button className="btn small ghost" disabled={progress.pending === progress.total} onClick={() => onSetReview(item.id, listed.map((d) => d.id), 'pending')}>
              <RotateCcw size={13} /> Reset
            </button>
          </div>
        )}
      </div>

      {tab === 'fov' ? (
        <div className="fov">
          <div className="fov-stage">
            <ImageViewer
              ref={viewerRef}
              src={item.url}
              width={result.image.width}
              height={result.image.height}
              detections={all}
              review={review}
              selectedId={selectedId}
              onSelect={(id) => select(id, false)}
              showRejected={showRejected}
            />
          </div>
          <Inspector
            item={item}
            progress={progress}
            selected={selected}
            onSetState={decide}
            showRejected={showRejected}
            onToggleRejected={() => setShowRejected((v) => !v)}
          />
        </div>
      ) : (
        <ConcentrationGrid
          item={item}
          detections={listed}
          selectedId={selectedId}
          onSelect={(id) => select(id, false)}
          onSetState={(id, state) => onSetReview(item.id, [id], state)}
          onOpenInFov={(id) => { setTab('fov'); setSelectedId(id); requestAnimationFrame(() => select(id, true)); }}
          gridRef={gridRef}
        />
      )}
    </div>
  );
}
