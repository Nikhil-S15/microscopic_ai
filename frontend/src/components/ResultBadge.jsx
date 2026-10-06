import React from 'react';
import { AlertTriangle, CheckCircle2, Clock3, Loader2, ShieldAlert, XCircle } from 'lucide-react';

/** Status chip for one sample. Every state has an icon + text, never colour alone. */
export default function ResultBadge({ item, metrics, size = 'md' }) {
  let tone; let Icon; let text;
  if (item.status === 'queued') { tone = 'neutral'; Icon = Clock3; text = 'Queued'; }
  else if (item.status === 'analyzing') { tone = 'info'; Icon = Loader2; text = 'Analysing'; }
  else if (item.status === 'error') { tone = 'error'; Icon = XCircle; text = 'Failed'; }
  else if (metrics?.detected) { tone = 'detected'; Icon = size === 'lg' ? ShieldAlert : AlertTriangle; text = 'Bacteria detected'; }
  else { tone = 'clear'; Icon = CheckCircle2; text = 'No bacteria detected'; }
  return (
    <span className={`badge ${tone} ${size}`}>
      <Icon size={size === 'lg' ? 18 : 13} className={item.status === 'analyzing' ? 'spin' : ''} aria-hidden="true" />
      {text}
    </span>
  );
}
