import { reviewProgress, reviewState, sampleMetrics } from './format';

function download(name, blob) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

const csvCell = (v) => {
  const s = v == null ? '' : String(v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
};
const toCsv = (rows) => rows.map((r) => r.map(csvCell).join(',')).join('\n');
const stamp = () => new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-');

export function exportDetectionsCsv(items) {
  const rows = [['sample', 'detection', 'x1', 'y1', 'x2', 'y2', 'confidence', 'review']];
  for (const it of items) {
    if (!it.result) continue;
    for (const d of it.result.detections) {
      rows.push([it.file.name, d.id + 1, ...d.bbox, d.confidence, reviewState(it, d.id)]);
    }
  }
  download(`detections_${stamp()}.csv`, new Blob([toCsv(rows)], { type: 'text/csv' }));
}

export function exportSummaryCsv(items) {
  const rows = [['sample', 'status', 'result', 'detections', 'mean_confidence', 'max_confidence', 'coverage', 'confirmed', 'rejected', 'pending', 'processing_s']];
  for (const it of items) {
    const m = sampleMetrics(it);
    const p = reviewProgress(it);
    rows.push([
      it.file.name, it.status,
      m ? (m.detected ? 'Bacteria detected' : 'No bacteria detected') : '',
      m?.count ?? '',
      m?.meanConfidence?.toFixed(4) ?? '',
      m?.maxConfidence?.toFixed(4) ?? '',
      m ? m.coverage.toFixed(6) : '',
      m ? p.confirmed : '', m ? p.rejected : '', m ? p.pending : '',
      it.result?.processing_time ?? '',
    ]);
  }
  download(`batch_summary_${stamp()}.csv`, new Blob([toCsv(rows)], { type: 'text/csv' }));
}

/** Render the image with every non-rejected box burned in, and download as PNG. */
export async function exportAnnotatedPng(item) {
  const img = new Image();
  img.src = item.url;
  await img.decode();
  const canvas = document.createElement('canvas');
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0);
  const lw = Math.max(2, Math.round(canvas.width / 600));
  ctx.font = `600 ${lw * 7}px Inter, system-ui, sans-serif`;
  for (const d of item.result.detections) {
    const state = reviewState(item, d.id);
    if (state === 'rejected') continue;
    const color = state === 'confirmed' ? '#22c55e' : '#06b6d4';
    const [x1, y1, x2, y2] = d.bbox;
    ctx.lineWidth = lw + 2;
    ctx.strokeStyle = 'rgba(4, 20, 40, 0.65)';
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.lineWidth = lw;
    ctx.strokeStyle = color;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const label = `#${d.id + 1} ${(d.confidence * 100).toFixed(0)}%`;
    const tw = ctx.measureText(label).width;
    const th = lw * 9;
    ctx.fillStyle = color;
    ctx.fillRect(x1, Math.max(0, y1 - th), tw + lw * 3, th);
    ctx.fillStyle = '#04121f';
    ctx.fillText(label, x1 + lw * 1.5, Math.max(th, y1) - lw * 2);
  }
  const blob = await new Promise((r) => canvas.toBlob(r, 'image/png'));
  download(`${item.file.name.replace(/\.[^.]+$/, '')}_annotated.png`, blob);
}
