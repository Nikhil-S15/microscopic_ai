import React, { forwardRef, useCallback, useEffect, useImperativeHandle, useLayoutEffect, useRef, useState } from 'react';
import { Maximize, Minus, Plus } from 'lucide-react';

const MAX_SCALE = 8;
const STATE_COLOR = { pending: '#22d3ee', confirmed: '#4ade80', rejected: '#94a3b8' };
const STATE_LABEL = { pending: 'Awaiting review', confirmed: 'Confirmed', rejected: 'Rejected' };

/**
 * Zoom/pan image stage with an SVG detection overlay.
 * Detection coordinates are natural image pixels; colour encodes review state.
 */
const ImageViewer = forwardRef(function ImageViewer(
  { src, width, height, detections, review, selectedId, onSelect, showRejected = true },
  ref,
) {
  const stageRef = useRef(null);
  const [view, setView] = useState(null);           // {s, x, y}
  const [fitted, setFitted] = useState(true);
  const [hover, setHover] = useState(null);         // {det, px, py}
  const drag = useRef(null);

  const fitView = useCallback(() => {
    const el = stageRef.current;
    if (!el) return null;
    const { clientWidth: cw, clientHeight: ch } = el;
    const s = Math.min(cw / width, ch / height);
    return { s, x: (cw - width * s) / 2, y: (ch - height * s) / 2 };
  }, [width, height]);

  useLayoutEffect(() => {
    setView(fitView());
    setFitted(true);
  }, [fitView, src]);

  useEffect(() => {
    const el = stageRef.current;
    if (!el) return undefined;
    const ro = new ResizeObserver(() => { if (fitted) setView(fitView()); });
    ro.observe(el);
    return () => ro.disconnect();
  }, [fitted, fitView]);

  const zoomAt = useCallback((factor, px, py) => {
    setView((v) => {
      if (!v) return v;
      const minS = (fitView()?.s ?? v.s) * 0.5;
      const s = Math.min(MAX_SCALE, Math.max(minS, v.s * factor));
      const k = s / v.s;
      return { s, x: px - (px - v.x) * k, y: py - (py - v.y) * k };
    });
    setFitted(false);
  }, [fitView]);

  const zoomCenter = (factor) => {
    const el = stageRef.current;
    zoomAt(factor, el.clientWidth / 2, el.clientHeight / 2);
  };

  const focusBox = useCallback((bbox) => {
    const el = stageRef.current;
    if (!el) return;
    const [x1, y1, x2, y2] = bbox;
    const cw = el.clientWidth, ch = el.clientHeight;
    const fit = fitView()?.s ?? 1;
    const s = Math.min(MAX_SCALE, Math.max(fit * 3, Math.min(cw, ch) / (Math.max(x2 - x1, y2 - y1) * 6)));
    setView({ s, x: cw / 2 - ((x1 + x2) / 2) * s, y: ch / 2 - ((y1 + y2) / 2) * s });
    setFitted(false);
  }, [fitView]);

  const fit = useCallback(() => { setView(fitView()); setFitted(true); }, [fitView]);
  useImperativeHandle(ref, () => ({ focusBox, fit }), [focusBox, fit]);

  useEffect(() => {
    const el = stageRef.current;
    if (!el) return undefined;
    const onWheel = (e) => {
      e.preventDefault();
      const r = el.getBoundingClientRect();
      zoomAt(Math.exp(-e.deltaY * 0.0015), e.clientX - r.left, e.clientY - r.top);
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, [zoomAt]);

  const onPointerDown = (e) => {
    if (e.button !== 0) return;
    drag.current = { sx: e.clientX, sy: e.clientY, vx: view.x, vy: view.y, moved: false, target: e.target };
    stageRef.current.setPointerCapture(e.pointerId);
  };
  const onPointerMove = (e) => {
    const d = drag.current;
    if (!d) return;
    const dx = e.clientX - d.sx, dy = e.clientY - d.sy;
    if (!d.moved && Math.hypot(dx, dy) > 3) { d.moved = true; setHover(null); }
    if (d.moved) {
      setView((v) => ({ ...v, x: d.vx + dx, y: d.vy + dy }));
      setFitted(false);
    }
  };
  const onPointerUp = (e) => {
    const d = drag.current;
    drag.current = null;
    if (d && !d.moved) {
      const id = d.target?.dataset?.det;
      onSelect(id != null ? Number(id) : null);
    }
    stageRef.current?.releasePointerCapture?.(e.pointerId);
  };

  const scale = view?.s ?? 1;
  const shown = showRejected ? detections : detections.filter((d) => review[d.id] !== 'rejected');

  return (
    <div className="viewer">
      <div
        ref={stageRef}
        className="stage"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerLeave={() => setHover(null)}
        onDoubleClick={(e) => {
          const r = stageRef.current.getBoundingClientRect();
          zoomAt(2, e.clientX - r.left, e.clientY - r.top);
        }}
      >
        {view && (
          <div className="stage-content" style={{ width, height, transform: `translate(${view.x}px, ${view.y}px) scale(${view.s})` }}>
            <img src={src} alt="Microscope field of view" draggable={false} width={width} height={height} />
            <svg className="overlay" viewBox={`0 0 ${width} ${height}`} width={width} height={height}>
              {shown.map((d) => {
                const [x1, y1, x2, y2] = d.bbox;
                const state = review[d.id] ?? 'pending';
                const color = STATE_COLOR[state];
                const isSel = selectedId === d.id;
                const isHover = hover?.det.id === d.id;
                const w = x2 - x1, h = y2 - y1;
                const sw = (isSel || isHover ? 3.5 : 2) / scale;
                const showTag = w * scale > 40 || isSel;
                const label = `#${d.id + 1} · ${Math.round(d.confidence * 100)}%`;
                return (
                  <g key={d.id} className={`det ${state} ${isSel ? 'selected' : ''}`}>
                    <rect x={x1} y={y1} width={w} height={h} className="halo" strokeWidth={sw + 2.5 / scale} />
                    <rect
                      x={x1} y={y1} width={w} height={h}
                      data-det={d.id}
                      className="box"
                      stroke={color}
                      strokeWidth={sw}
                      strokeDasharray={state === 'rejected' ? `${6 / scale} ${4 / scale}` : undefined}
                      fill={isSel || isHover ? `${color}26` : 'transparent'}
                      onPointerEnter={(e) => {
                        if (drag.current?.moved) return;
                        const r = stageRef.current.getBoundingClientRect();
                        setHover({ det: d, px: e.clientX - r.left, py: e.clientY - r.top });
                      }}
                      onPointerLeave={() => setHover((h0) => (h0?.det.id === d.id ? null : h0))}
                    />
                    {showTag && state !== 'rejected' && (
                      <g transform={`translate(${x1} ${y1}) scale(${1 / scale})`} className="tag-g">
                        <rect x={-1} y={-19} width={label.length * 6.4 + 10} height={18} rx={4} fill={color} />
                        <text x={5} y={-6}>{label}</text>
                      </g>
                    )}
                  </g>
                );
              })}
            </svg>
          </div>
        )}

        {hover && (
          <div className="tooltip floating" role="tooltip" style={{ left: hover.px + 14, top: hover.py + 14 }}>
            <span className="tt-title">Detection #{hover.det.id + 1}</span>
            <span className="tt-row"><span>Confidence</span><strong>{(hover.det.confidence * 100).toFixed(1)}%</strong></span>
            <span className="tt-row"><span>Size</span><strong>{hover.det.bbox[2] - hover.det.bbox[0]} × {hover.det.bbox[3] - hover.det.bbox[1]} px</strong></span>
            <span className="tt-row"><span>Status</span><strong>{STATE_LABEL[review[hover.det.id] ?? 'pending']}</strong></span>
          </div>
        )}
      </div>

      <div className="zoom-controls" role="group" aria-label="Zoom">
        <button className="icon-btn" onClick={() => zoomCenter(1.4)} aria-label="Zoom in" title="Zoom in"><Plus size={15} /></button>
        <span className="zoom-level">{Math.round(scale * 100)}%</span>
        <button className="icon-btn" onClick={() => zoomCenter(1 / 1.4)} aria-label="Zoom out" title="Zoom out"><Minus size={15} /></button>
        <span className="divider" />
        <button className="icon-btn" onClick={fit} aria-label="Fit to screen" title="Fit to screen (Esc)"><Maximize size={15} /></button>
        <button
          className="icon-btn"
          onClick={() => { const el = stageRef.current; zoomAt(1 / scale, el.clientWidth / 2, el.clientHeight / 2); }}
          aria-label="Actual size" title="Actual size (100%)"
        >
          <span className="one-to-one">1:1</span>
        </button>
      </div>
    </div>
  );
});

export default ImageViewer;
