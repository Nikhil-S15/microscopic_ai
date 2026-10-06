import React from 'react';
import { Cpu, ImagePlus, Layers, ScanSearch, ServerCrash, ShieldCheck } from 'lucide-react';

export default function EmptyState({ model, modelError, onBrowse }) {
  return (
    <div className="empty">
      <span className="eyebrow"><ScanSearch size={14} /> AI-assisted microscopy</span>
      <h1>Detect bacteria in microscope images, <span className="grad-text">in seconds</span></h1>
      <p className="lede">
        Upload one or many fields of view. Every image is scanned, each detection is located
        and scored, and you confirm the result before export.
      </p>

      {modelError && !model ? (
        <div className="alert error">
          <ServerCrash size={18} />
          <span><strong>Analysis server offline.</strong> Start the backend on port 8000 — this page reconnects automatically.</span>
        </div>
      ) : (
        <button className="upload-hero glass" onClick={onBrowse} disabled={!model}>
          <span className="upload-icon"><ImagePlus size={26} strokeWidth={1.8} /></span>
          <strong>Upload samples</strong>
          <span>Drag and drop images anywhere, or click to browse</span>
          <span className="formats">JPG · PNG · TIFF · BMP — up to {model?.limits.max_image_mb ?? 40} MB each</span>
        </button>
      )}

      <ul className="trust">
        <li><Layers size={16} /> Batch processing</li>
        <li><ShieldCheck size={16} /> Expert review of every detection</li>
        <li><Cpu size={16} /> Runs locally{model ? ` on ${model.device.toUpperCase()}` : ''}</li>
      </ul>
    </div>
  );
}
