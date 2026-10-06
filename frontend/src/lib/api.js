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
