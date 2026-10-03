// App.jsx — 2-Step Pipeline
//   Step 1: Upload → classify bacteria species (7-class Keras model)
//   Step 2: User sees classification result → auto-runs bounding box detection

import React, { useState, useEffect } from 'react';
import { Camera, RotateCcw, AlertCircle, Sparkles, ChevronRight, FlaskConical } from 'lucide-react';
import ResultsDisplay from './components/ResultsDisplay';
import ClassificationResult from './components/ClassificationResult';
import './App.css';

const API_URL = 'http://localhost:8000';

// Pipeline stages
const STAGE = {
  IDLE:        'idle',
  CLASSIFYING: 'classifying',
  CLASSIFIED:  'classified',
  DETECTING:   'detecting',
  DONE:        'done',
};

function App() {
  const [stage,          setStage]          = useState(STAGE.IDLE);
  const [image,          setImage]          = useState(null);
  const [imageFile,      setImageFile]      = useState(null);
  const [classifyResult, setClassifyResult] = useState(null);
  const [detectResult,   setDetectResult]   = useState(null);
  const [error,          setError]          = useState(null);

  // Review state
  const [confirmedDetections, setConfirmedDetections] = useState([]);
  const [removedDetections,   setRemovedDetections]   = useState([]);
  const [selectedDetection,   setSelectedDetection]   = useState(null);

  const settings = {
    confidence: 0.1,
    nms:        0.3,
    tta:        true,
    top_k:      3,
  };

  // ── Step 1: Classify ────────────────────────────────────────────────────────
  const runClassify = async (file) => {
    setStage(STAGE.CLASSIFYING);
    setError(null);

    try {
      const form = new FormData();
      form.append('file', file);

      const res = await fetch(`${API_URL}/api/v1/classify`, {
        method: 'POST',
        body: form,
      });

      if (!res.ok) {
        let msg = 'Classification failed';
        try { const d = await res.json(); msg = d.detail || msg; } catch {}
        throw new Error(msg);
      }

      const data = await res.json();
      setClassifyResult(data);
      setStage(STAGE.CLASSIFIED);

      // Auto-run detection after 1.2s so user can see classification first
      setTimeout(() => runDetect(file), 1200);

    } catch (err) {
      setError(err.message);
      setStage(STAGE.IDLE);
    }
  };

  // ── Step 2: Detect ──────────────────────────────────────────────────────────
  const runDetect = async (file) => {
    setStage(STAGE.DETECTING);

    try {
      const form   = new FormData();
      form.append('file', file);

      const params = new URLSearchParams({
        confidence_threshold: settings.confidence.toString(),
        nms_threshold:        settings.nms.toString(),
        use_tta:              settings.tta.toString(),
        top_k:                settings.top_k.toString(),
        ...(classifyResult?.predicted_class
          ? { species_name: classifyResult.predicted_class }
          : {}),
      });

      const res = await fetch(`${API_URL}/api/v1/detect?${params}`, {
        method: 'POST',
        body: form,
      });

      if (!res.ok) {
        let msg = 'Detection failed';
        try { const d = await res.json(); msg = d.detail || d.message || msg; } catch {}
        throw new Error(msg);
      }

      const data = await res.json();
      if (!data.success) throw new Error(data.message || 'Detection failed');

      setDetectResult(data);
      setStage(STAGE.DONE);

    } catch (err) {
      setError(err.message);
      // Stay on classified stage so user can still see classification
      setStage(STAGE.CLASSIFIED);
    }
  };

  // ── Handle upload ───────────────────────────────────────────────────────────
  const handleImageUpload = async (file) => {
    if (!file.type.startsWith('image/')) {
      setError('Please upload an image file (JPEG, PNG, etc.)');
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      setError('Image size should be less than 10MB');
      return;
    }

    // Reset state
    setConfirmedDetections([]);
    setRemovedDetections([]);
    setSelectedDetection(null);
    setClassifyResult(null);
    setDetectResult(null);
    setError(null);

    setImageFile(file);
    setImage(URL.createObjectURL(file));

    await runClassify(file);
  };

  const handleReset = () => {
    if (image) URL.revokeObjectURL(image);
    setImage(null);
    setImageFile(null);
    setClassifyResult(null);
    setDetectResult(null);
    setError(null);
    setStage(STAGE.IDLE);
    setConfirmedDetections([]);
    setRemovedDetections([]);
    setSelectedDetection(null);
  };

  // ── Review handlers ─────────────────────────────────────────────────────────
  const confirmDetection = (index) => {
    setConfirmedDetections(prev => prev.includes(index) ? prev : [...prev, index]);
    setRemovedDetections(prev => prev.filter(i => i !== index));
  };

  const removeDetection = (index) => {
    setRemovedDetections(prev => prev.includes(index) ? prev : [...prev, index]);
    setConfirmedDetections(prev => prev.filter(i => i !== index));
    setSelectedDetection(prev => prev?.index === index ? null : prev);
  };

  const handleDetectionClick = (detection, index) => {
    setSelectedDetection({ ...detection, index });
  };

  // ── Render ──────────────────────────────────────────────────────────────────
  const isLoading = stage === STAGE.CLASSIFYING || stage === STAGE.DETECTING;

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="header-content">
          <FlaskConical className="header-icon" />
          <div>
            <h1 className="header-title">Bacteria Detection System</h1>
            <p className="header-subtitle">✨ AI-Powered Microscopic Analysis</p>
          </div>
        </div>

        {/* Pipeline progress indicator */}
        {stage !== STAGE.IDLE && (
          <div className="pipeline-steps">
            <div className={`pipeline-step ${stage !== STAGE.IDLE ? 'done-or-active' : ''} ${stage === STAGE.CLASSIFYING ? 'active' : ''}`}>
              <span className="step-num">1</span>
              <span className="step-label">Identify Species</span>
            </div>
            <ChevronRight size={16} className="step-arrow" />
            <div className={`pipeline-step ${stage === STAGE.DETECTING || stage === STAGE.DONE ? 'done-or-active' : ''} ${stage === STAGE.DETECTING ? 'active' : ''}`}>
              <span className="step-num">2</span>
              <span className="step-label">Detect &amp; Locate</span>
            </div>
          </div>
        )}
      </header>

      {error && (
        <div className="error-banner">
          <AlertCircle size={24} />
          <div className="error-content">
            <p className="error-title">Error</p>
            <p className="error-message">{error}</p>
          </div>
          <button className="error-close" onClick={() => setError(null)}>×</button>
        </div>
      )}

      <main className="main-content">
        <div className="content-wrapper">

          {/* ── IDLE: Upload screen ────────────────────────────────────────── */}
          {stage === STAGE.IDLE && (
            <div className="upload-section">
              <div className="upload-card">
                <div className="upload-header">
                  <h2>Upload Microscope Image</h2>
                  <p>AI will identify the bacteria species, then locate each cell with bounding boxes</p>
                </div>

                {/* 2-step explainer */}
                <div className="pipeline-explainer">
                  <div className="pipe-step">
                    <div className="pipe-icon step1">🔬</div>
                    <div>
                      <strong>Step 1 — Identify</strong>
                      <p>7-class model names the species with confidence scores</p>
                    </div>
                  </div>
                  <div className="pipe-arrow">→</div>
                  <div className="pipe-step">
                    <div className="pipe-icon step2">📦</div>
                    <div>
                      <strong>Step 2 — Locate</strong>
                      <p>Sliding-window CNN draws bounding boxes around each bacterium</p>
                    </div>
                  </div>
                </div>

                <div className="upload-area">
                  <input
                    type="file" id="file-upload" accept="image/*"
                    onChange={e => { if (e.target.files?.[0]) handleImageUpload(e.target.files[0]); }}
                    style={{ display: 'none' }}
                  />
                  <label htmlFor="file-upload" className="upload-label">
                    <div className="upload-icon"><Camera size={64} /></div>
                    <p className="upload-title">Click to select image</p>
                    <p className="upload-subtitle">or drag and drop here</p>
                    <p className="upload-formats">Supported: JPG, PNG, WebP (max 10MB)</p>
                  </label>
                </div>
              </div>
            </div>
          )}

          {/* ── CLASSIFYING: Step 1 spinner ───────────────────────────────── */}
          {stage === STAGE.CLASSIFYING && (
            <div className="loading-container">
              <div className="step-badge">Step 1 of 2</div>
              <div className="spinner spinner-classify"></div>
              <h2 className="loading-title">Identifying Bacteria Species…</h2>
              <p className="loading-subtitle">
                Running 7-class classification on<br />
                <strong>{imageFile?.name}</strong>
              </p>
              {image && <img src={image} alt="preview" className="loading-preview" />}
            </div>
          )}

          {/* ── CLASSIFIED: Show result briefly while detect starts ────────── */}
          {(stage === STAGE.CLASSIFIED || stage === STAGE.DETECTING || stage === STAGE.DONE) && classifyResult && (
            <ClassificationResult
              result={classifyResult}
              detecting={stage === STAGE.DETECTING}
              detectionDone={stage === STAGE.DONE}
              imageName={imageFile?.name}
            />
          )}

          {/* ── DONE: Full detection results below classification ──────────── */}
          {stage === STAGE.DONE && image && detectResult && (
            <div className="detect-section">
              <div className="detect-header-bar">
                <h3>🗺 Step 2 — Bounding Box Detection</h3>
                <span className="detect-count">{detectResult.total_detections} cells found</span>
                <button className="reset-btn" onClick={handleReset}>
                  <RotateCcw size={16} /> New Image
                </button>
              </div>
              <ResultsDisplay
                image={image}
                results={detectResult}
                onReset={handleReset}
                selectedDetection={selectedDetection}
                onDetectionClick={handleDetectionClick}
                confirmedDetections={confirmedDetections}
                removedDetections={removedDetections}
                onConfirmDetection={confirmDetection}
                onRemoveDetection={removeDetection}
              />
            </div>
          )}

        </div>
      </main>

      <footer className="app-footer">
        <div className="footer-content">
          <p>Bacteria Detection System v2.0 • 2-Step AI Pipeline</p>
          <p className="footer-note">Identify species → Locate individual cells</p>
        </div>
      </footer>
    </div>
  );
}

export default App;