// Requests go through the Vite proxy (/api → http://localhost:8000).
async function parseError(res, fallback) {
  try {
    const body = await res.json();
    return body.detail || fallback;
  } catch {
    return fallback;
  }
}

export async function fetchModelInfo() {
  const res = await fetch('/api/v2/model');
  if (!res.ok) throw new Error(await parseError(res, 'Model service unavailable'));
  return res.json();
}

/**
 * Precision / recall / F1 / detection count per confidence threshold for
 * already-analysed images. images: [{ request_id, ground_truth: [[x1,y1,x2,y2]] }].
 */
export async function runThresholdAnalysis(thresholds, images, { signal } = {}) {
  let res;
  try {
    res = await fetch('/api/v2/threshold-analysis', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ thresholds, images }),
      signal,
    });
  } catch (err) {
    if (err.name === 'TypeError') throw new Error('Cannot reach the analysis server');
    throw err;
  }
  if (!res.ok) throw new Error(await parseError(res, `Threshold analysis failed (${res.status})`));
  return res.json();
}

/** Download the saved masks of analysed samples as one ZIP. items: [{ request_id, name }]. */
export async function downloadMasksZip(items) {
  const res = await fetch('/api/v2/masks/export', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ items }),
  });
  if (!res.ok) throw new Error(await parseError(res, `Mask export failed (${res.status})`));
  return res.blob();
}

/**
 * Analyse one image. Polls /progress while the request runs and reports 0–1
 * through onProgress. Resolves with the analysis result.
 */
export async function analyzeImage(file, settings, { onProgress, signal } = {}) {
  const requestId = crypto.randomUUID();
  const form = new FormData();
  form.append('file', file);
  form.append('request_id', requestId);
  form.append('sensitivity', String(settings.sensitivity));
  form.append('noise_gate', String(settings.noiseGate));
  if (settings.threshold != null) form.append('threshold', String(settings.threshold));
  form.append('box_screening', String(settings.boxScreening));
  form.append('noise_cutoff', String(settings.noiseCutoff));
  form.append('sam_verify', String(settings.samVerify));
  form.append('mask_threshold', String(settings.maskThreshold));
  form.append('min_coverage', String(settings.minCoverage));

  let stopped = false;
  const poll = async () => {
    while (!stopped) {
      await new Promise((r) => setTimeout(r, 450));
      if (stopped) break;
      try {
        const res = await fetch(`/api/v2/progress/${requestId}`, { signal });
        if (res.ok) {
          const p = await res.json();
          if (!stopped && p.status === 'analyzing') onProgress?.(p.progress);
        }
      } catch {
        /* polling is best-effort */
      }
    }
  };
  poll();

  try {
    const res = await fetch('/api/v2/analyze', { method: 'POST', body: form, signal });
    if (!res.ok) throw new Error(await parseError(res, `Analysis failed (${res.status})`));
    return await res.json();
  } catch (err) {
    if (err.name === 'TypeError') throw new Error('Cannot reach the analysis server');
    throw err;
  } finally {
    stopped = true;
  }
}
