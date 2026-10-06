import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { UploadCloud } from 'lucide-react';
import { analyzeImage, fetchModelInfo } from './lib/api';
import TopBar from './components/TopBar';
import SettingsDrawer from './components/SettingsDrawer';
import Dashboard from './components/Dashboard';
import ReviewView from './components/ReviewView';
import EmptyState from './components/EmptyState';

const IMAGE_ACCEPT = { 'image/*': ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp', '.jfif'] };

const settingsKey = (s) => JSON.stringify([s.sensitivity, s.noiseGate]);

function readTheme() {
  try { return localStorage.getItem('theme') || 'system'; } catch { return 'system'; }
}

export default function App() {
  const [model, setModel] = useState(null);
  const [modelError, setModelError] = useState(null);
  const [items, setItems] = useState([]);
  const [settings, setSettings] = useState({ sensitivity: 0, noiseGate: true });
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [running, setRunning] = useState(true);
  const [view, setView] = useState('dashboard');
  const [selectedId, setSelectedId] = useState(null);
  const [theme, setTheme] = useState(readTheme);
  const busy = useRef(false);
  const abortRef = useRef(null);

  // ── theme ──────────────────────────────────────────────────────────────
  useEffect(() => {
    const root = document.documentElement;
    if (theme === 'system') root.removeAttribute('data-theme');
    else root.setAttribute('data-theme', theme);
    try { localStorage.setItem('theme', theme); } catch { /* ignore */ }
  }, [theme]);

  // ── model info (retry until the backend is up) ─────────────────────────
  useEffect(() => {
    let cancelled = false;
    let timer;
    const load = async () => {
      try {
        const info = await fetchModelInfo();
        if (cancelled) return;
        setModel(info);
        setModelError(null);
        if (!info.noise_gate) setSettings((s) => ({ ...s, noiseGate: false }));
      } catch (e) {
        if (cancelled) return;
        setModelError(e.message);
        timer = setTimeout(load, 3000);
      }
    };
    load();
    return () => { cancelled = true; clearTimeout(timer); };
  }, []);

  const update = useCallback((id, patch) => {
    setItems((prev) => prev.map((it) => (it.id === id ? { ...it, ...(typeof patch === 'function' ? patch(it) : patch) } : it)));
  }, []);

  // ── queue: one image at a time (the server uses every core) ────────────
  useEffect(() => {
    if (!running || busy.current || !model) return;
    const next = items.find((it) => it.status === 'queued');
    if (!next) return;
    busy.current = true;
    const controller = new AbortController();
    abortRef.current = controller;
    const usedSettings = settings;
    update(next.id, { status: 'analyzing', progress: 0, error: null, startedAt: Date.now() });
    analyzeImage(next.file, usedSettings, {
      signal: controller.signal,
      onProgress: (p) => update(next.id, (it) => (it.status === 'analyzing' ? { progress: p } : {})),
    })
      .then((result) => update(next.id, {
        status: 'done', progress: 1, result, review: {}, settingsKey: settingsKey(usedSettings),
      }))
      .catch((err) => update(next.id, err.name === 'AbortError'
        ? { status: 'queued', progress: 0 }
        : { status: 'error', error: err.message }))
      .finally(() => {
        busy.current = false;
        abortRef.current = null;
        setItems((prev) => [...prev]);   // re-run this effect for the next item
      });
  }, [items, running, settings, model, update]);

  // ── actions ────────────────────────────────────────────────────────────
  const addFiles = useCallback((files) => {
    const fresh = files.map((file) => ({
      id: crypto.randomUUID(),
      file,
      url: URL.createObjectURL(file),
      status: 'queued',
      progress: 0,
      result: null,
      error: null,
      review: {},
    }));
    if (!fresh.length) return;
    setItems((prev) => [...prev, ...fresh]);
    setRunning(true);
  }, []);

  const { getRootProps, getInputProps, isDragActive, open: browse } = useDropzone({
    onDrop: addFiles,
    accept: IMAGE_ACCEPT,
    maxSize: (model?.limits.max_image_mb ?? 40) * 1024 * 1024,
    multiple: true,
    noClick: true,
    noKeyboard: true,
    disabled: !model,
  });

  const removeItem = useCallback((id) => {
    setItems((prev) => {
      const it = prev.find((x) => x.id === id);
      if (it?.status === 'analyzing') abortRef.current?.abort();
      if (it) URL.revokeObjectURL(it.url);
      return prev.filter((x) => x.id !== id);
    });
  }, []);

  const clearAll = useCallback(() => {
    abortRef.current?.abort();
    setItems((prev) => { prev.forEach((it) => URL.revokeObjectURL(it.url)); return []; });
    setSelectedId(null);
    setView('dashboard');
  }, []);

  const retry = useCallback((id) => update(id, { status: 'queued', progress: 0, error: null }), [update]);

  const reanalyzeAll = useCallback(() => {
    setItems((prev) => prev.map((it) => (it.status === 'analyzing' ? it : { ...it, status: 'queued', progress: 0, error: null })));
    setRunning(true);
    setSettingsOpen(false);
  }, []);

  const toggleRunning = useCallback(() => {
    setRunning((r) => {
      if (r) abortRef.current?.abort();
      return !r;
    });
  }, []);

  /** Set review state ('confirmed' | 'rejected' | 'pending') for detections of one sample. */
  const setReview = useCallback((id, detIds, state) => update(id, (it) => {
    const review = { ...it.review };
    for (const d of detIds) {
      if (state === 'pending') delete review[d]; else review[d] = state;
    }
    return { review };
  }), [update]);

  const openReview = useCallback((id) => { setSelectedId(id); setView('review'); }, []);

  const doneItems = items.filter((it) => it.status === 'done');
  useEffect(() => {
    if (view === 'review' && !doneItems.some((it) => it.id === selectedId)) {
      if (doneItems.length) setSelectedId(doneItems[0].id); else setView('dashboard');
    }
  }, [view, selectedId, doneItems]);

  const staleCount = doneItems.filter((it) => it.settingsKey !== settingsKey(settings)).length;
  const selected = doneItems.find((it) => it.id === selectedId);

  return (
    <div {...getRootProps({ className: 'app' })}>
      <input {...getInputProps()} aria-label="Add microscope images" />
      <div className="backdrop" aria-hidden="true" />
      <TopBar
        model={model}
        modelError={modelError}
        theme={theme}
        onTheme={setTheme}
        view={view}
        onView={setView}
        canReview={doneItems.length > 0}
        hasItems={items.length > 0}
        onAdd={browse}
        onSettings={() => setSettingsOpen(true)}
        settingsDirty={staleCount > 0}
      />

      <main className="main" id="main">
        {items.length === 0 ? (
          <EmptyState model={model} modelError={modelError} onBrowse={browse} />
        ) : view === 'review' && selected ? (
          <ReviewView
            key={selected.id}
            item={selected}
            items={doneItems}
            model={model}
            onSelect={setSelectedId}
            onSetReview={setReview}
            onBack={() => setView('dashboard')}
          />
        ) : (
          <Dashboard
            items={items}
            model={model}
            settings={settings}
            running={running}
            staleCount={staleCount}
            onAdd={browse}
            onToggleRunning={toggleRunning}
            onReanalyze={reanalyzeAll}
            onOpen={openReview}
            onRemove={removeItem}
            onRetry={retry}
            onClear={clearAll}
          />
        )}
      </main>

      {model && (
        <SettingsDrawer
          open={settingsOpen}
          onClose={() => setSettingsOpen(false)}
          model={model}
          settings={settings}
          onChange={setSettings}
          staleCount={staleCount}
          onReanalyze={reanalyzeAll}
        />
      )}

      {isDragActive && (
        <div className="drop-overlay" aria-hidden="true">
          <div className="drop-card">
            <span className="drop-icon"><UploadCloud size={30} /></span>
            <strong>Drop to add samples</strong>
            <span>Images are queued and analysed automatically</span>
          </div>
        </div>
      )}
    </div>
  );
}
