/** Confidence-threshold sweep: threshold list, best-threshold selection, wording. */

export const SWEEP_DEFAULTS = { start: 0.95, end: 0.98, step: 0.01 };
export const STEP_OPTIONS = [0.0025, 0.005, 0.01, 0.02];
export const MAX_THRESHOLDS = 101;
/** F1-scores this close (0.25 points) count as tied; secondary criteria then decide. */
export const F1_TIE = 0.0025;

const round4 = (v) => Math.round(v * 10000) / 10000;
const close = (a, b) => Math.abs(a - b) < 1e-9;

/** Thresholds from start to end (inclusive) in `step` increments, or an error message. */
export function buildThresholds(start, end, step, min = 0.5) {
  if (![start, end, step].every(Number.isFinite)) return { error: 'Enter numeric values' };
  if (start < min || end >= 1) return { error: `Thresholds must be between ${min} and 0.999` };
  if (end < start) return { error: 'End must not be below start' };
  if (step <= 0) return { error: 'Step must be positive' };
  const n = Math.floor((end - start) / step + 1e-9) + 1;
  if (n > MAX_THRESHOLDS) return { error: `Too many thresholds (${n}); use a larger step` };
  const values = Array.from({ length: n }, (_, i) => round4(start + i * step));
  if (end - values[values.length - 1] > 1e-9) values.push(round4(end));
  return { values };
}

/** Decimal places needed to show every threshold exactly (min 2). */
export function thresholdDigits(values) {
  return Math.max(2, ...values.map((v) => (String(v).split('.')[1] || '').length));
}

const pct = (v) => (v == null ? '—' : `${(v * 100).toFixed(1)}%`);
const pts = (d) => `${Math.abs(d * 100).toFixed(1)} points`;

/**
 * Pick the best threshold.
 *   1. highest F1-score (balances precision and recall);
 *   2. among thresholds within F1_TIE of it: higher precision + recall combined,
 *      then higher precision,
 *   3. then the detection count closest to the number of labelled objects,
 *   4. then the lower threshold.
 * Returns { index, tied: [indices], rule } or null when no F1 can be computed.
 */
export function pickBest(rows, objects) {
  const scored = rows.map((r, i) => ({ ...r, i })).filter((r) => r.f1 != null);
  if (!scored.length) return null;
  const top = Math.max(...scored.map((r) => r.f1));
  const tied = scored.filter((r) => r.f1 >= top - F1_TIE - 1e-9);
  const both = (r) => (r.precision ?? 0) + (r.recall ?? 0);
  const off = (r) => Math.abs(r.detections - objects);
  const ranked = [...tied].sort((a, b) =>
    both(b) - both(a) ||
    (b.precision ?? -1) - (a.precision ?? -1) ||
    off(a) - off(b) ||
    a.threshold - b.threshold);
  const best = ranked[0];
  let rule = 'f1';
  if (tied.length > 1) {
    const second = ranked[1];
    if (!close(both(best), both(second))) rule = 'balance';
    else if (!close(best.precision ?? -1, second.precision ?? -1)) rule = 'precision';
    else if (off(best) !== off(second)) rule = 'count';
    else rule = 'lowest';
  }
  return { index: best.i, tied: tied.map((r) => r.i), rule };
}

/** Plain-language justification for the selected threshold (2–3 sentences). */
export function justify(rows, best, objects, digits) {
  const fmt = (t) => t.toFixed(digits);
  const b = rows[best.index];
  const parts = [];
  if (best.tied.length === 1) {
    parts.push(`${fmt(b.threshold)} has the highest F1-score (${pct(b.f1)}), giving the best balance between precision (${pct(b.precision)}) and recall (${pct(b.recall)}).`);
  } else {
    const list = best.tied.map((i) => fmt(rows[i].threshold));
    const names = list.length === 2 ? list.join(' and ') : `${list.slice(0, -1).join(', ')} and ${list[list.length - 1]}`;
    const why = {
      balance: `it has the highest precision and recall combined (${pct(b.precision)} precision, ${pct(b.recall)} recall)`,
      precision: `with the same combined precision and recall it has the higher precision (${pct(b.precision)}), so fewer detections are false alarms`,
      count: `its ${b.detections} detections are closest to the ${objects} labelled objects`,
      lowest: `their results are identical (${pct(b.precision)} precision, ${pct(b.recall)} recall), so the lowest is kept`,
    }[best.rule];
    const same = best.tied.every((i) => close(rows[i].f1, b.f1));
    parts.push(`${names} have ${same ? 'the same' : 'near-identical'} F1-score${same ? ` (${pct(b.f1)})` : `s (within ${+(F1_TIE * 100).toFixed(2)} points)`}. ${fmt(b.threshold)} is selected because ${why}.`);
  }
  // trade-off against the neighbouring higher threshold, if any
  const next = rows[best.index + 1];
  if (next && next.recall != null && b.recall != null) {
    const dr = next.recall - b.recall;
    const dp = (next.precision ?? 0) - (b.precision ?? 0);
    if (dr < -1e-9) {
      parts.push(`Raising it to ${fmt(next.threshold)} would ${dp > 1e-9 ? `gain ${pts(dp)} of precision but ` : ''}lose ${pts(dr)} of recall.`);
    } else if (dp < -1e-9) {
      parts.push(`Raising it to ${fmt(next.threshold)} would lower precision by ${pts(dp)}.`);
    }
  } else if (!next && best.index > 0) {
    parts.push('It is the highest threshold tested — extend the range upwards to check whether the F1-score keeps rising.');
  }
  return parts.join(' ');
}
