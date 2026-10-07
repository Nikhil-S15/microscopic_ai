import React from 'react';
import { ChartSpline, LayoutDashboard, Microscope, Monitor, Moon, Plus, ScanSearch, SlidersHorizontal, Sun } from 'lucide-react';

const THEMES = [
  { id: 'light', icon: Sun, label: 'Light' },
  { id: 'system', icon: Monitor, label: 'System' },
  { id: 'dark', icon: Moon, label: 'Dark' },
];

export default function TopBar({
  model, modelError, theme, onTheme, view, onView, canReview, hasItems, onAdd, onSettings, settingsDirty,
}) {
  const status = model
    ? { tone: 'ok', text: 'AI model online', title: `${model.architecture} · ${model.device.toUpperCase()}` }
    : modelError
      ? { tone: 'error', text: 'Server offline', title: 'Analysis server is not reachable — retrying' }
      : { tone: 'pending', text: 'Connecting', title: 'Connecting to the analysis server' };
  const nextTheme = THEMES[(THEMES.findIndex((t) => t.id === theme) + 1) % THEMES.length];
  const ThemeIcon = THEMES.find((t) => t.id === theme)?.icon ?? Monitor;

  return (
    <header className="topbar">
      <div className="brand">
        <span className="brand-mark" aria-hidden="true"><Microscope size={17} strokeWidth={2.2} /></span>
        <span className="brand-text">
          <span className="brand-name">Microscopy AI</span>
          <span className="brand-sub">Bacteria detection</span>
        </span>
      </div>

      {hasItems && (
        <nav className="nav" aria-label="View">
          <button className={view === 'dashboard' ? 'active' : ''} onClick={() => onView('dashboard')}>
            <LayoutDashboard size={15} /> <span>Dashboard</span>
          </button>
          <button className={view === 'review' ? 'active' : ''} onClick={() => onView('review')} disabled={!canReview}>
            <ScanSearch size={15} /> <span>Review</span>
          </button>
          <button className={view === 'thresholds' ? 'active' : ''} onClick={() => onView('thresholds')} disabled={!canReview}>
            <ChartSpline size={15} /> <span>Thresholds</span>
          </button>
        </nav>
      )}

      <div className="topbar-right">
        <span className={`live ${status.tone}`} title={status.title}>
          <span className="live-dot" aria-hidden="true" />{status.text}
        </span>
        <button className="icon-btn" onClick={() => onTheme(nextTheme.id)} title={`Theme: ${theme} — switch to ${nextTheme.label.toLowerCase()}`} aria-label="Change theme">
          <ThemeIcon size={16} />
        </button>
        <button className="icon-btn" onClick={onSettings} disabled={!model} title="Analysis settings" aria-label="Analysis settings">
          <SlidersHorizontal size={16} />
          {settingsDirty && <span className="badge-dot" aria-hidden="true" />}
        </button>
        {hasItems && (
          <button className="btn primary" onClick={onAdd} disabled={!model}>
            <Plus size={16} /> <span className="hide-sm">Add samples</span>
          </button>
        )}
      </div>
    </header>
  );
}
