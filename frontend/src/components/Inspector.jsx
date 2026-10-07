import React from 'react';
import { Check, MousePointerClick, RotateCcw, X } from 'lucide-react';
import CropView from './CropView';
import { pct } from '../lib/format';

const STATE_COLOR = { pending: '#22d3ee', confirmed: '#4ade80', rejected: '#94a3b8' };
const STATE_LABEL = { pending: 'Awaiting review', confirmed: 'Confirmed', rejected: 'Rejected' };

export default function Inspector({
  item, progress, selected, onSetState, showRejected, onToggleRejected, showRemoved, onToggleRemoved, showMasks, onToggleMasks,
}) {
  const { result, review } = item;
  const sc = result.screening;
  const removedNoise = sc?.removed_noise ?? 0;
  const removedMask = sc?.removed_mask ?? 0;
  const hasMasks = result.detections.some((d) => d.mask) || (result.removed ?? []).some((d) => d.mask);
  const state = selected ? review[selected.id] ?? 'pending' : null;
  const reviewed = progress.total - progress.pending;

  return (
    <aside className="inspector">
      <section>
        <div className="insp-head">
          <h3>Review progress</h3>
          <span className="muted small">{reviewed} of {progress.total}</span>
        </div>
        <div className="progress ok"><span style={{ width: `${progress.total ? (reviewed / progress.total) * 100 : 0}%` }} /></div>
        <dl className="tally">
          <div className="confirmed"><dt><Check size={12} /> Confirmed</dt><dd>{progress.confirmed}</dd></div>
          <div className="rejected"><dt><X size={12} /> Rejected</dt><dd>{progress.rejected}</dd></div>
          <div><dt>Pending</dt><dd>{progress.pending}</dd></div>
        </dl>
      </section>

      <section className="insp-selected">
        {selected ? (
          <>
            <div className="insp-head">
              <h3>Detection #{selected.id + 1}</h3>
              <span className={`state-pill ${state}`}>{STATE_LABEL[state]}</span>
            </div>
            <div className="insp-crop">
              <CropView src={item.url} width={result.image.width} height={result.image.height} bbox={selected.bbox} color={STATE_COLOR[state]} context={3} />
            </div>
            <dl className="facts">
              <dt>Confidence</dt>
              <dd className="conf-dd">
                <span className="conf-bar"><span style={{ width: `${selected.confidence * 100}%` }} /></span>
                {pct(selected.confidence, 1)}
              </dd>
              <dt>Size</dt><dd>{selected.bbox[2] - selected.bbox[0]} × {selected.bbox[3] - selected.bbox[1]} px</dd>
              <dt>Position</dt><dd>x {selected.bbox[0]}, y {selected.bbox[1]}</dd>
              {selected.noise_score != null && <><dt>Noise probability</dt><dd>{pct(selected.noise_score, 1)}</dd></>}
              {selected.mask && <><dt>Mask coverage</dt><dd>{pct(selected.mask.coverage, 1)}</dd></>}
            </dl>
            <div className="decide">
              <button className={`btn decide-btn confirm ${state === 'confirmed' ? 'on' : ''}`} onClick={() => onSetState(selected.id, 'confirmed')}>
                <Check size={15} /> Confirm <kbd>C</kbd>
              </button>
              <button className={`btn decide-btn reject ${state === 'rejected' ? 'on' : ''}`} onClick={() => onSetState(selected.id, 'rejected')}>
                <X size={15} /> Reject <kbd>R</kbd>
              </button>
              {state !== 'pending' && (
                <button className="icon-btn" onClick={() => onSetState(selected.id, 'pending')} title="Undo (U)" aria-label="Undo decision">
                  <RotateCcw size={14} />
                </button>
              )}
            </div>
          </>
        ) : (
          <div className="insp-hint">
            <span className="hint-icon"><MousePointerClick size={18} /></span>
            <span>Select a box to inspect it, or press <kbd>→</kbd> to step through detections.</span>
          </div>
        )}
      </section>

      {sc && (
        <section>
          <h3 className="legend-title">Screening</h3>
          <ol className="funnel">
            <li><span>Boxes detected</span><strong>{sc.boxes_detected}</strong></li>
            <li className={sc.box_screening ? '' : 'off'}><span>Removed as noise</span><strong>{sc.box_screening ? (removedNoise ? `−${removedNoise}` : '0') : 'off'}</strong></li>
            <li className={sc.sam_verify ? '' : 'off'}><span>Removed by mask check</span><strong>{sc.sam_verify ? (removedMask ? `−${removedMask}` : '0') : 'off'}</strong></li>
            <li className="total"><span>Kept for review</span><strong>{sc.kept}</strong></li>
          </ol>
          <p className="fine-print">
            Threshold {sc.threshold ?? 'model defaults'}
            {sc.box_screening && <> · noise &gt; {sc.noise_cutoff}</>}
            {sc.sam_verify && <> · mask coverage ≥ {Math.round(sc.min_coverage * 100)}%</>}
          </p>
        </section>
      )}

      <section>
        <h3 className="legend-title">Legend</h3>
        <ul className="legend">
          <li><span className="lg-box pending" /> Awaiting review</li>
          <li><span className="lg-box confirmed" /> Confirmed</li>
          <li>
            <span className="lg-box rejected" /> Rejected
            <button className="link-btn" onClick={onToggleRejected}>{showRejected ? 'Hide' : 'Show'}</button>
          </li>
          {hasMasks && (
            <li>
              <span className="lg-mask" /> Object mask
              <button className="link-btn" onClick={onToggleMasks}>{showMasks ? 'Hide' : 'Show'}</button>
            </li>
          )}
          {removedNoise + removedMask > 0 && (
            <li>
              <span className="lg-box removed" /> Removed by screening ({removedNoise + removedMask})
              <button className="link-btn" onClick={onToggleRemoved}>{showRemoved ? 'Hide' : 'Show'}</button>
            </li>
          )}
        </ul>
      </section>
    </aside>
  );
}
