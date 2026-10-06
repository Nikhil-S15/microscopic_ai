export const pct = (v, digits = 0) =>
  v == null ? '—' : `${(v * 100).toFixed(digits)}%`;

export const seconds = (s) =>
  s == null ? '—' : s < 60 ? `${s.toFixed(1)} s` : `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;

export const num = (n) => (n == null ? '—' : n.toLocaleString());

/** Review state of one detection: 'pending' | 'confirmed' | 'rejected'. */
export const reviewState = (item, detId) => item.review[detId] ?? 'pending';

/** {total, confirmed, rejected, pending} for an analysed item. */
export function reviewProgress(item) {
  const total = item.result?.detections.length ?? 0;
  let confirmed = 0;
  let rejected = 0;
  for (const s of Object.values(item.review)) {
    if (s === 'confirmed') confirmed += 1;
    else if (s === 'rejected') rejected += 1;
  }
  return { total, confirmed, rejected, pending: total - confirmed - rejected };
}

const mean = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : null);

/**
 * Detection metrics for an analysed item, after review: rejected detections
 * are excluded, so the result reflects the reviewer's decisions.
 */
export function sampleMetrics(item, highConfidence = 0.95) {
  if (!item.result) return null;
  const kept = item.result.detections.filter((d) => item.review[d.id] !== 'rejected');
  const confs = kept.map((d) => d.confidence);
  const { width, height } = item.result.image;
  const area = kept.reduce((a, d) => a + (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]), 0);
  return {
    count: kept.length,
    detected: kept.length > 0,
    meanConfidence: mean(confs),
    maxConfidence: confs.length ? Math.max(...confs) : null,
    highShare: confs.length ? confs.filter((c) => c >= highConfidence).length / confs.length : null,
    // boxes rarely overlap after merging; capped at 1 for safety
    coverage: Math.min(1, area / (width * height)),
    confidences: confs,
  };
}
