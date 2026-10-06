import React, { useEffect } from 'react';
import { Check, X } from 'lucide-react';
import CropView from './CropView';

const STATE_COLOR = { pending: '#22d3ee', confirmed: '#4ade80', rejected: '#94a3b8' };

/** Digital concentration: one context crop per detection, confirmed or rejected in place. */
export default function ConcentrationGrid({ item, detections, selectedId, onSelect, onSetState, onOpenInFov, gridRef }) {
  const { result, review } = item;

  useEffect(() => {
    if (selectedId == null) return;
    gridRef.current?.querySelector(`[data-id="${selectedId}"]`)?.scrollIntoView({ block: 'nearest' });
  }, [selectedId, gridRef]);

  if (!detections.length) {
    return <div className="grid-empty"><span className="muted">No detections match this filter.</span></div>;
  }

  return (
    <div className="tile-grid" ref={gridRef} role="listbox" aria-label="Detections" aria-activedescendant={selectedId != null ? `tile-${selectedId}` : undefined}>
      {detections.map((d) => {
        const state = review[d.id] ?? 'pending';
        return (
          <div
            key={d.id}
            id={`tile-${d.id}`}
            data-id={d.id}
            role="option"
            aria-selected={selectedId === d.id}
            className={`tile ${state} ${selectedId === d.id ? 'selected' : ''}`}
            onClick={() => onSelect(d.id)}
            onDoubleClick={() => onOpenInFov(d.id)}
            title="Double-click to locate in the full field of view"
          >
            <span className="tile-media">
              <CropView src={item.url} width={result.image.width} height={result.image.height} bbox={d.bbox} color={STATE_COLOR[state]} dimmed={state === 'rejected'} />
              <span className="tile-num">#{d.id + 1}</span>
              {state !== 'pending' && (
                <span className={`tile-state ${state}`} aria-label={state}>
                  {state === 'confirmed' ? <Check size={12} strokeWidth={3} /> : <X size={12} strokeWidth={3} />}
                </span>
              )}
            </span>
            <span className="tile-foot">
              <span className="tile-conf">
                <span className="conf-bar"><span style={{ width: `${d.confidence * 100}%` }} /></span>
                <span>{Math.round(d.confidence * 100)}%</span>
              </span>
              <span className="tile-actions">
                <button
                  className={`tile-btn confirm ${state === 'confirmed' ? 'on' : ''}`}
                  onClick={(e) => { e.stopPropagation(); onSetState(d.id, state === 'confirmed' ? 'pending' : 'confirmed'); }}
                  aria-label={`Confirm detection ${d.id + 1}`}
                  aria-pressed={state === 'confirmed'}
                  title="Confirm (C)"
                >
                  <Check size={14} />
                </button>
                <button
                  className={`tile-btn reject ${state === 'rejected' ? 'on' : ''}`}
                  onClick={(e) => { e.stopPropagation(); onSetState(d.id, state === 'rejected' ? 'pending' : 'rejected'); }}
                  aria-label={`Reject detection ${d.id + 1}`}
                  aria-pressed={state === 'rejected'}
                  title="Reject (R)"
                >
                  <X size={14} />
                </button>
              </span>
            </span>
          </div>
        );
      })}
    </div>
  );
}
