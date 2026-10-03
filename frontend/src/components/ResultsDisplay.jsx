// ResultsDisplay.jsx — Complete with image enhancement + Compact side panel actions

import React, { useRef, useState, useEffect, useCallback } from 'react';
import {
  Download, RotateCcw, Eye, Filter, CheckCircle, XCircle, Sparkles,
  Contrast, Sun, RotateCcw as RotateIcon, ZoomIn, Sliders, Shield, X
} from 'lucide-react';
import BoundingBoxCanvas from './BoundingBoxCanvas';

// ─────────────────────────────────────────────────────────────────────────────
// IMAGE ENHANCEMENT CLASS (Built-in)
// ─────────────────────────────────────────────────────────────────────────────

class ImageEnhancer {
  static upscaleNearest(imageData, scale) {
    const width = imageData.width;
    const height = imageData.height;
    const data = imageData.data;
    
    const newWidth = width * scale;
    const newHeight = height * scale;
    const output = new Uint8ClampedArray(newWidth * newHeight * 4);
    
    for (let y = 0; y < newHeight; y++) {
      for (let x = 0; x < newWidth; x++) {
        const srcX = Math.floor(x / scale);
        const srcY = Math.floor(y / scale);
        const srcIdx = (srcY * width + srcX) * 4;
        const dstIdx = (y * newWidth + x) * 4;
        
        output[dstIdx] = data[srcIdx];
        output[dstIdx + 1] = data[srcIdx + 1];
        output[dstIdx + 2] = data[srcIdx + 2];
        output[dstIdx + 3] = data[srcIdx + 3];
      }
    }
    
    return new ImageData(output, newWidth, newHeight);
  }

  static upscaleBilinear(imageData, scale) {
    const width = imageData.width;
    const height = imageData.height;
    const data = imageData.data;
    
    const newWidth = width * scale;
    const newHeight = height * scale;
    const output = new Uint8ClampedArray(newWidth * newHeight * 4);
    
    for (let y = 0; y < newHeight; y++) {
      for (let x = 0; x < newWidth; x++) {
        const srcX = x / scale;
        const srcY = y / scale;
        
        const x1 = Math.floor(srcX);
        const y1 = Math.floor(srcY);
        const x2 = Math.min(x1 + 1, width - 1);
        const y2 = Math.min(y1 + 1, height - 1);
        
        const fx = srcX - x1;
        const fy = srcY - y1;
        
        const idx11 = (y1 * width + x1) * 4;
        const idx12 = (y1 * width + x2) * 4;
        const idx21 = (y2 * width + x1) * 4;
        const idx22 = (y2 * width + x2) * 4;
        
        for (let c = 0; c < 3; c++) {
          const v11 = data[idx11 + c];
          const v12 = data[idx12 + c];
          const v21 = data[idx21 + c];
          const v22 = data[idx22 + c];
          
          const top = v11 * (1 - fx) + v12 * fx;
          const bottom = v21 * (1 - fx) + v22 * fx;
          const value = top * (1 - fy) + bottom * fy;
          
          const dstIdx = (y * newWidth + x) * 4 + c;
          output[dstIdx] = Math.min(255, Math.max(0, Math.round(value)));
        }
        
        const dstIdx = (y * newWidth + x) * 4 + 3;
        output[dstIdx] = 255;
      }
    }
    
    return new ImageData(output, newWidth, newHeight);
  }

  static cubicInterpolation(p0, p1, p2, p3, t) {
    const a0 = p2 - p0;
    const a1 = p0 - p2;
    const a2 = p3 - p1;
    const a3 = p1 - p3;
    
    const b0 = p1;
    const b1 = a1 / 2;
    const b2 = a2 / 2;
    const b3 = (a0 + a3) / 2;
    
    return b0 + b1 * t + b2 * t * t + b3 * t * t * t;
  }

  static upscaleBicubic(imageData, scale) {
    const width = imageData.width;
    const height = imageData.height;
    const data = imageData.data;
    
    const newWidth = width * scale;
    const newHeight = height * scale;
    const output = new Uint8ClampedArray(newWidth * newHeight * 4);
    
    for (let y = 0; y < newHeight; y++) {
      for (let x = 0; x < newWidth; x++) {
        const srcX = x / scale;
        const srcY = y / scale;
        
        const x1 = Math.floor(srcX);
        const y1 = Math.floor(srcY);
        
        const x0 = Math.max(0, x1 - 1);
        const x2 = Math.min(width - 1, x1 + 1);
        const x3 = Math.min(width - 1, x1 + 2);
        const y0 = Math.max(0, y1 - 1);
        const y2 = Math.min(height - 1, y1 + 1);
        const y3 = Math.min(height - 1, y1 + 2);
        
        const fx = srcX - x1;
        const fy = srcY - y1;
        
        for (let c = 0; c < 3; c++) {
          const rowValues = [];
          
          for (let ry of [y0, y1, y2, y3]) {
            const colValues = [];
            for (let rx of [x0, x1, x2, x3]) {
              const idx = (ry * width + rx) * 4 + c;
              colValues.push(data[idx]);
            }
            const value = this.cubicInterpolation(colValues[0], colValues[1], colValues[2], colValues[3], fx);
            rowValues.push(value);
          }
          
          const value = this.cubicInterpolation(rowValues[0], rowValues[1], rowValues[2], rowValues[3], fy);
          const dstIdx = (y * newWidth + x) * 4 + c;
          output[dstIdx] = Math.min(255, Math.max(0, Math.round(value)));
        }
        
        const dstIdx = (y * newWidth + x) * 4 + 3;
        output[dstIdx] = 255;
      }
    }
    
    return new ImageData(output, newWidth, newHeight);
  }

  static applyPixelation(imageData, blockSize) {
    const width = imageData.width;
    const height = imageData.height;
    const data = imageData.data;
    const output = new Uint8ClampedArray(data.length);
    
    for (let y = 0; y < height; y += blockSize) {
      for (let x = 0; x < width; x += blockSize) {
        let sumR = 0, sumG = 0, sumB = 0;
        let count = 0;
        
        const blockW = Math.min(blockSize, width - x);
        const blockH = Math.min(blockSize, height - y);
        
        for (let by = 0; by < blockH; by++) {
          for (let bx = 0; bx < blockW; bx++) {
            const idx = ((y + by) * width + (x + bx)) * 4;
            sumR += data[idx];
            sumG += data[idx + 1];
            sumB += data[idx + 2];
            count++;
          }
        }
        
        const avgR = sumR / count;
        const avgG = sumG / count;
        const avgB = sumB / count;
        
        for (let by = 0; by < blockH; by++) {
          for (let bx = 0; bx < blockW; bx++) {
            const idx = ((y + by) * width + (x + bx)) * 4;
            output[idx] = avgR;
            output[idx + 1] = avgG;
            output[idx + 2] = avgB;
            output[idx + 3] = data[idx + 3];
          }
        }
      }
    }
    
    return new ImageData(output, width, height);
  }

  static applySharpen(imageData, intensity = 1.0) {
    const width = imageData.width;
    const height = imageData.height;
    const data = imageData.data;
    const output = new Uint8ClampedArray(data.length);
    
    const kernel = [
      0, -intensity, 0,
      -intensity, 1 + (intensity * 4), -intensity,
      0, -intensity, 0
    ];
    
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        for (let c = 0; c < 3; c++) {
          let sum = 0;
          let kidx = 0;
          
          for (let ky = -1; ky <= 1; ky++) {
            for (let kx = -1; kx <= 1; kx++) {
              const idx = ((y + ky) * width + (x + kx)) * 4 + c;
              sum += data[idx] * kernel[kidx++];
            }
          }
          
          const idx = (y * width + x) * 4 + c;
          output[idx] = Math.min(255, Math.max(0, sum));
          output[idx + 3] = data[idx + 3];
        }
      }
    }
    
    for (let i = 0; i < data.length; i++) {
      if (output[i] === 0 && data[i] !== 0) {
        output[i] = data[i];
      }
    }
    
    return new ImageData(output, width, height);
  }

  static applyEdgeEnhancement(imageData, strength = 0.5) {
    const width = imageData.width;
    const height = imageData.height;
    const data = imageData.data;
    const output = new Uint8ClampedArray(data.length);
    
    const sobelX = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
    const sobelY = [-1, -2, -1, 0, 0, 0, 1, 2, 1];
    
    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        for (let c = 0; c < 3; c++) {
          let gx = 0, gy = 0;
          let kidx = 0;
          
          for (let ky = -1; ky <= 1; ky++) {
            for (let kx = -1; kx <= 1; kx++) {
              const idx = ((y + ky) * width + (x + kx)) * 4 + c;
              gx += data[idx] * sobelX[kidx];
              gy += data[idx] * sobelY[kidx];
              kidx++;
            }
          }
          
          const edge = Math.sqrt(gx * gx + gy * gy);
          const idx = (y * width + x) * 4 + c;
          output[idx] = Math.min(255, Math.max(0, data[idx] + edge * strength));
          output[idx + 3] = data[idx + 3];
        }
      }
    }
    
    return new ImageData(output, width, height);
  }

  static applyMedianFilter(imageData, radius = 1) {
    const width = imageData.width;
    const height = imageData.height;
    const data = imageData.data;
    const output = new Uint8ClampedArray(data.length);
    const halfKernel = radius;
    
    for (let y = halfKernel; y < height - halfKernel; y++) {
      for (let x = halfKernel; x < width - halfKernel; x++) {
        for (let c = 0; c < 3; c++) {
          const values = [];
          
          for (let ky = -halfKernel; ky <= halfKernel; ky++) {
            for (let kx = -halfKernel; kx <= halfKernel; kx++) {
              const idx = ((y + ky) * width + (x + kx)) * 4 + c;
              values.push(data[idx]);
            }
          }
          
          values.sort((a, b) => a - b);
          const median = values[Math.floor(values.length / 2)];
          const idx = (y * width + x) * 4 + c;
          output[idx] = median;
          output[idx + 3] = data[idx + 3];
        }
      }
    }
    
    return new ImageData(output, width, height);
  }

  static async enhanceImage(imageElement, settings) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    canvas.width = imageElement.width;
    canvas.height = imageElement.height;
    ctx.drawImage(imageElement, 0, 0);
    
    let imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    
    if (settings.denoise > 0) {
      const radius = Math.max(1, Math.floor(settings.denoise * 3));
      imageData = this.applyMedianFilter(imageData, radius);
    }
    
    if (settings.pixelation > 0) {
      const blockSize = Math.max(2, Math.floor(16 / settings.pixelation));
      imageData = this.applyPixelation(imageData, blockSize);
    }
    
    if (settings.edgeEnhancement > 0) {
      imageData = this.applyEdgeEnhancement(imageData, settings.edgeEnhancement);
    }
    
    if (settings.sharpen > 0) {
      imageData = this.applySharpen(imageData, settings.sharpen);
    }
    
    const contrast = settings.contrast || 1.0;
    const brightness = settings.brightness || 1.0;
    const data = imageData.data;
    
    for (let i = 0; i < data.length; i += 4) {
      let r = ((data[i] / 255 - 0.5) * contrast + 0.5) * 255 * brightness;
      let g = ((data[i+1] / 255 - 0.5) * contrast + 0.5) * 255 * brightness;
      let b = ((data[i+2] / 255 - 0.5) * contrast + 0.5) * 255 * brightness;
      
      data[i] = Math.min(255, Math.max(0, r));
      data[i+1] = Math.min(255, Math.max(0, g));
      data[i+2] = Math.min(255, Math.max(0, b));
    }
    
    ctx.putImageData(imageData, 0, 0);
    
    if (settings.upscale > 1) {
      let upscaled;
      switch (settings.interpolation) {
        case 'nearest':
          upscaled = this.upscaleNearest(imageData, settings.upscale);
          break;
        case 'bilinear':
          upscaled = this.upscaleBilinear(imageData, settings.upscale);
          break;
        case 'bicubic':
          upscaled = this.upscaleBicubic(imageData, settings.upscale);
          break;
        default:
          upscaled = this.upscaleBilinear(imageData, settings.upscale);
      }
      
      const upscaledCanvas = document.createElement('canvas');
      upscaledCanvas.width = upscaled.width;
      upscaledCanvas.height = upscaled.height;
      upscaledCanvas.getContext('2d').putImageData(upscaled, 0, 0);
      return upscaledCanvas;
    }
    
    return canvas;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// IMAGE ENHANCER MODAL COMPONENT (Built-in)
// ─────────────────────────────────────────────────────────────────────────────

function ImageEnhancerModal({ originalImage, onEnhanced, onClose }) {
  const [settings, setSettings] = useState({
    contrast: 1.2,
    brightness: 1.1,
    sharpen: 0.3,
    denoise: 0.2,
    edgeEnhancement: 0.2,
    pixelation: 0,
    interpolation: 'bilinear',
    upscale: 1
  });
  const [isProcessing, setIsProcessing] = useState(false);
  const [previewImage, setPreviewImage] = useState(originalImage);
  const [showOriginal, setShowOriginal] = useState(false);
  
  const applyEnhancements = useCallback(async () => {
    setIsProcessing(true);
    
    const img = new Image();
    img.src = originalImage;
    
    await new Promise((resolve) => { img.onload = resolve; });
    
    try {
      const resultCanvas = await ImageEnhancer.enhanceImage(img, settings);
      const enhancedDataUrl = resultCanvas.toDataURL('image/jpeg', 0.95);
      setPreviewImage(enhancedDataUrl);
    } catch (error) {
      console.error('Enhancement failed:', error);
    } finally {
      setIsProcessing(false);
    }
  }, [originalImage, settings]);
  
  useEffect(() => {
    const timer = setTimeout(() => {
      applyEnhancements();
    }, 300);
    return () => clearTimeout(timer);
  }, [applyEnhancements]);
  
  const handleApply = () => {
    onEnhanced(previewImage);
    onClose();
  };
  
  const resetSettings = () => {
    setSettings({
      contrast: 1.0,
      brightness: 1.0,
      sharpen: 0,
      denoise: 0,
      edgeEnhancement: 0,
      pixelation: 0,
      interpolation: 'bilinear',
      upscale: 1
    });
    setPreviewImage(originalImage);
  };
  
  return (
    <div className="enhancer-modal-overlay" onClick={onClose}>
      <div className="enhancer-modal" onClick={(e) => e.stopPropagation()}>
        <div className="enhancer-header">
          <h3><Sparkles size={20} /> Enhance Image Quality</h3>
          <button onClick={onClose} className="enhancer-close"><X size={20} /></button>
        </div>
        
        <div className="enhancer-body">
          <div className="enhancer-controls">
            <div className="enhancer-control-group">
              <label><Contrast size={14} /> Contrast</label>
              <input type="range" min="0.5" max="2.0" step="0.01"
                value={settings.contrast}
                onChange={(e) => setSettings({...settings, contrast: parseFloat(e.target.value)})}
              />
              <span>{settings.contrast.toFixed(2)}x</span>
            </div>
            
            <div className="enhancer-control-group">
              <label><Sun size={14} /> Brightness</label>
              <input type="range" min="0.5" max="2.0" step="0.01"
                value={settings.brightness}
                onChange={(e) => setSettings({...settings, brightness: parseFloat(e.target.value)})}
              />
              <span>{settings.brightness.toFixed(2)}x</span>
            </div>
            
            <div className="enhancer-control-group">
              <label>🔪 Sharpness</label>
              <input type="range" min="0" max="1.5" step="0.01"
                value={settings.sharpen}
                onChange={(e) => setSettings({...settings, sharpen: parseFloat(e.target.value)})}
              />
              <span>{settings.sharpen.toFixed(2)}</span>
              <small>Enhances edges and details</small>
            </div>
            
            <div className="enhancer-control-group">
              <label><Shield size={14} /> Noise Reduction</label>
              <input type="range" min="0" max="1" step="0.05"
                value={settings.denoise}
                onChange={(e) => setSettings({...settings, denoise: parseFloat(e.target.value)})}
              />
              <span>{settings.denoise.toFixed(2)}</span>
              <small>Removes noise/grain</small>
            </div>
            
            <div className="enhancer-control-group">
              <label>🔍 Edge Enhancement</label>
              <input type="range" min="0" max="1" step="0.05"
                value={settings.edgeEnhancement}
                onChange={(e) => setSettings({...settings, edgeEnhancement: parseFloat(e.target.value)})}
              />
              <span>{settings.edgeEnhancement.toFixed(2)}</span>
              <small>Makes cell boundaries visible</small>
            </div>
            
            <div className="enhancer-control-group">
              <label><Sliders size={14} /> Pixelation Effect</label>
              <input type="range" min="0" max="1" step="0.05"
                value={settings.pixelation}
                onChange={(e) => setSettings({...settings, pixelation: parseFloat(e.target.value)})}
              />
              <span>{settings.pixelation.toFixed(2)}</span>
              <small>Creates pixelated block effect</small>
            </div>
            
            <div className="enhancer-control-group">
              <label>🎨 Interpolation Method</label>
              <select
                value={settings.interpolation}
                onChange={(e) => setSettings({...settings, interpolation: e.target.value})}
              >
                <option value="nearest">Nearest-Neighbor (Pixelated/Sharp)</option>
                <option value="bilinear">Bilinear (Smooth)</option>
                <option value="bicubic">Bicubic (Very Smooth)</option>
              </select>
              <small>Nearest-neighbor preserves sharp edges</small>
            </div>
            
            <div className="enhancer-control-group">
              <label><ZoomIn size={14} /> Upscale</label>
              <select
                value={settings.upscale}
                onChange={(e) => setSettings({...settings, upscale: parseInt(e.target.value)})}
              >
                <option value={1}>No upscale</option>
                <option value={2}>2x (HD)</option>
                <option value={3}>3x</option>
                <option value={4}>4x (4K)</option>
              </select>
            </div>
            
            <div className="enhancer-buttons">
              <button onClick={resetSettings} className="enhancer-btn-secondary">
                <RotateIcon size={14} /> Reset
              </button>
              <button onClick={handleApply} className="enhancer-btn-primary" disabled={isProcessing}>
                {isProcessing ? 'Processing...' : 'Apply Enhancement'}
              </button>
            </div>
          </div>
          
          <div className="enhancer-preview">
            <div className="enhancer-preview-header">
              <button className={`preview-toggle ${!showOriginal ? 'active' : ''}`}
                onClick={() => setShowOriginal(false)}>Enhanced</button>
              <button className={`preview-toggle ${showOriginal ? 'active' : ''}`}
                onClick={() => setShowOriginal(true)}>Original</button>
            </div>
            <div className="enhancer-preview-image">
              {isProcessing && <div className="processing-overlay">Enhancing...</div>}
              <img src={showOriginal ? originalImage : previewImage} alt="Preview" />
            </div>
            <div className="enhancer-info">
              <small>💡 Tip: Use "Nearest-Neighbor" + "Edge Enhancement" for sharp cell boundaries</small>
            </div>
          </div>
        </div>
      </div>
      
      <style>{`
        .enhancer-modal-overlay {
          position: fixed; top: 0; left: 0; right: 0; bottom: 0;
          background: rgba(0,0,0,0.8); z-index: 2000;
          display: flex; align-items: center; justify-content: center;
        }
        .enhancer-modal {
          background: white; border-radius: 16px; width: 90%; max-width: 1100px;
          max-height: 90vh; overflow: auto; box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        .enhancer-header {
          display: flex; justify-content: space-between; align-items: center;
          padding: 20px 24px; border-bottom: 1px solid #e5e7eb;
        }
        .enhancer-header h3 { display: flex; align-items: center; gap: 8px; margin: 0; }
        .enhancer-close { background: none; border: none; font-size: 24px; cursor: pointer; color: #6b7280; }
        .enhancer-body { display: flex; padding: 24px; gap: 24px; flex-wrap: wrap; }
        .enhancer-controls { width: 320px; display: flex; flex-direction: column; gap: 16px; }
        .enhancer-control-group { display: flex; flex-direction: column; gap: 6px; }
        .enhancer-control-group label { display: flex; align-items: center; gap: 6px; font-size: 13px; font-weight: 600; }
        .enhancer-control-group input[type="range"] { width: 100%; }
        .enhancer-control-group select { padding: 6px; border-radius: 6px; border: 1px solid #d1d5db; }
        .enhancer-control-group small { font-size: 10px; color: #6b7280; }
        .enhancer-buttons { display: flex; gap: 12px; margin-top: 8px; }
        .enhancer-btn-primary, .enhancer-btn-secondary {
          flex: 1; padding: 10px; border-radius: 6px; font-weight: 600;
          display: flex; align-items: center; justify-content: center; gap: 6px; cursor: pointer;
        }
        .enhancer-btn-primary { background: #3b82f6; color: white; border: none; }
        .enhancer-btn-primary:disabled { opacity: 0.6; cursor: not-allowed; }
        .enhancer-btn-secondary { background: #f3f4f6; color: #374151; border: 1px solid #d1d5db; }
        .enhancer-preview { flex: 1; min-width: 300px; }
        .enhancer-preview-header { display: flex; gap: 8px; margin-bottom: 12px; }
        .preview-toggle { padding: 6px 16px; border: 1px solid #d1d5db; background: white; border-radius: 20px; cursor: pointer; }
        .preview-toggle.active { background: #3b82f6; color: white; border-color: #3b82f6; }
        .enhancer-preview-image { position: relative; border: 1px solid #e5e7eb; border-radius: 8px; overflow: hidden; background: #f9fafb; }
        .enhancer-preview-image img { width: 100%; height: auto; display: block; }
        .processing-overlay { position: absolute; top: 0; left: 0; right: 0; bottom: 0;
          background: rgba(0,0,0,0.5); display: flex; align-items: center; justify-content: center;
          color: white; font-weight: 500; backdrop-filter: blur(4px); }
        .enhancer-info { margin-top: 12px; padding: 8px; background: #f0fdf4; border-radius: 6px; text-align: center; }
      `}</style>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// DETECTION CROPPER with interpolation support
// ─────────────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────────────
// DETECTION CROPPER with larger surrounding context
// ─────────────────────────────────────────────────────────────────────────────

function DetectionCropper({ image, detection, interpolation = 'bilinear' }) {
  const containerRef = useRef(null);
  const canvasRef = useRef(null);
  const [imgLoaded, setImgLoaded] = useState(false);

  useEffect(() => {
    const container = containerRef.current;
    const canvas = canvasRef.current;
    if (!container || !canvas) return;

    const ctx = canvas.getContext('2d');
    const img = new Image();
    
    img.crossOrigin = "Anonymous";
    
    img.onload = () => {
      setImgLoaded(true);
      
      if (interpolation === 'nearest') {
        ctx.imageSmoothingEnabled = false;
      } else {
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = interpolation === 'bicubic' ? 'high' : 'medium';
      }
      
      const [x1, y1, x2, y2] = detection.bbox;
      const bW = x2 - x1;
      const bH = y2 - y1;

      // 🔥 INCREASED PADDING for more surrounding context
      // Now showing 3x the detection size for better context
      const paddingScale = 1.5; // Show 1.5x the detection size on each side
      const padX = Math.max(bW * paddingScale, 60); // Minimum 60px padding
      const padY = Math.max(bH * paddingScale, 60);

      // Calculate crop area with generous padding
      let srcX = Math.max(0, Math.min(img.width, x1 - padX));
      let srcY = Math.max(0, Math.min(img.height, y1 - padY));
      let srcX2 = Math.min(img.width, Math.max(0, x2 + padX));
      let srcY2 = Math.min(img.height, Math.max(0, y2 + padY));
      
      // Ensure minimum crop size
      if (srcX2 - srcX < 100) {
        const center = (srcX + srcX2) / 2;
        srcX = Math.max(0, center - 50);
        srcX2 = Math.min(img.width, center + 50);
      }
      if (srcY2 - srcY < 100) {
        const center = (srcY + srcY2) / 2;
        srcY = Math.max(0, center - 50);
        srcY2 = Math.min(img.height, center + 50);
      }
      
      let srcW = srcX2 - srcX;
      let srcH = srcY2 - srcY;

      const cssSize = 220; // Slightly larger display size
      const dpr = window.devicePixelRatio || 2;
      const px = Math.round(cssSize * dpr);
      
      canvas.width = px;
      canvas.height = px;
      canvas.style.width = `${cssSize}px`;
      canvas.style.height = `${cssSize}px`;
      
      // Calculate scaling to fit canvas while maintaining aspect ratio
      const scale = Math.min(px / srcW, px / srcH);
      const drawW = srcW * scale;
      const drawH = srcH * scale;
      const dstX = (px - drawW) / 2;
      const dstY = (px - drawH) / 2;
      
      // Draw background
      ctx.fillStyle = '#0f172a';
      ctx.fillRect(0, 0, px, px);
      
      // Draw the image
      ctx.drawImage(img, srcX, srcY, srcW, srcH, dstX, dstY, drawW, drawH);
      
      // Draw bounding box with position relative to the crop
      const bx = dstX + (x1 - srcX) * scale;
      const by = dstY + (y1 - srcY) * scale;
      const bw = bW * scale;
      const bh = bH * scale;
      
      // Only draw if the bounding box is reasonably within the canvas
      if (bx > -50 && by > -50 && bx + bw < px + 50 && by + bh < px + 50) {
        const color = detection.class_id === 0 ? '#10b981' : '#f59e0b';
        const glowColor = detection.class_id === 0 ? 'rgba(16, 185, 129, 0.4)' : 'rgba(245, 158, 11, 0.4)';
        
        // Draw glow/shadow
        ctx.save();
        ctx.shadowColor = glowColor;
        ctx.shadowBlur = 8 * dpr;
        ctx.strokeStyle = color;
        ctx.lineWidth = Math.max(2, 2.5 * dpr);
        ctx.strokeRect(bx, by, bw, bh);
        ctx.restore();
        
        // Draw inner white border for contrast
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.8)';
        ctx.lineWidth = Math.max(1, 1.5 * dpr);
        ctx.strokeRect(bx + 1, by + 1, bw - 2, bh - 2);
        
        // Draw corner markers
        const cs = Math.min(12 * dpr, bw * 0.15, bh * 0.15);
        ctx.strokeStyle = color;
        ctx.lineWidth = Math.max(2, 2.5 * dpr);
        
        // Top-left corner
        ctx.beginPath();
        ctx.moveTo(bx, by + cs);
        ctx.lineTo(bx, by);
        ctx.lineTo(bx + cs, by);
        ctx.stroke();
        
        // Top-right corner
        ctx.beginPath();
        ctx.moveTo(bx + bw - cs, by);
        ctx.lineTo(bx + bw, by);
        ctx.lineTo(bx + bw, by + cs);
        ctx.stroke();
        
        // Bottom-left corner
        ctx.beginPath();
        ctx.moveTo(bx, by + bh - cs);
        ctx.lineTo(bx, by + bh);
        ctx.lineTo(bx + cs, by + bh);
        ctx.stroke();
        
        // Bottom-right corner
        ctx.beginPath();
        ctx.moveTo(bx + bw - cs, by + bh);
        ctx.lineTo(bx + bw, by + bh);
        ctx.lineTo(bx + bw, by + bh - cs);
        ctx.stroke();
      }
    };
    
    img.src = image;
  }, [image, detection, interpolation]);

  return (
    <div ref={containerRef} style={{ 
      width: '100%', height: '100%', 
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      backgroundColor: '#0f172a', position: 'relative'
    }}>
      {!imgLoaded && <div style={{ color: '#94a3b8', fontSize: '12px' }}>Loading...</div>}
      <canvas ref={canvasRef} style={{ display: 'block', width: '100%', height: '100%', objectFit: 'contain' }} />
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN RESULTS DISPLAY COMPONENT
// ─────────────────────────────────────────────────────────────────────────────

function ResultsDisplay({
  image, results, onReset,
  selectedDetection, onDetectionClick,
  confirmedDetections, removedDetections,
  onConfirmDetection, onRemoveDetection,
}) {
  const canvasRef = useRef(null);
  const [activeTab, setActiveTab] = useState('visualization');
  const [cursorIndex, setCursorIndex] = useState(0);
  const [showEnhancer, setShowEnhancer] = useState(false);
  const [currentImage, setCurrentImage] = useState(image);
  const [isEnhanced, setIsEnhanced] = useState(false);

  // ONE filter pass
  const visible = results.detections
    .map((det, idx) => ({ ...det, _origIdx: idx }))
    .filter(det => !removedDetections.includes(det._origIdx));

  const safeIdx = visible.length === 0 ? 0 : Math.min(cursorIndex, visible.length - 1);
  useEffect(() => {
    if (safeIdx !== cursorIndex) setCursorIndex(safeIdx);
  }, [safeIdx, cursorIndex]);

  const confirmCurrent = useCallback(() => {
    if (!visible[safeIdx]) return;
    onConfirmDetection(visible[safeIdx]._origIdx);
    if (safeIdx < visible.length - 1) setCursorIndex(safeIdx + 1);
  }, [visible, safeIdx, onConfirmDetection]);

  const removeCurrent = useCallback(() => {
    if (!visible[safeIdx]) return;
    onRemoveDetection(visible[safeIdx]._origIdx);
  }, [visible, safeIdx, onRemoveDetection]);

  useEffect(() => {
    const handler = e => {
      if (activeTab !== 'concentration') return;
      if (['INPUT', 'TEXTAREA'].includes(e.target.tagName)) return;
      if (e.key === 'c' || e.key === 'C') { e.preventDefault(); confirmCurrent(); }
      if (e.key === 'r' || e.key === 'R') { e.preventDefault(); removeCurrent(); }
      if (e.key === 'ArrowRight') { e.preventDefault(); setCursorIndex(i => Math.min(i + 1, visible.length - 1)); }
      if (e.key === 'ArrowLeft')  { e.preventDefault(); setCursorIndex(i => Math.max(i - 1, 0)); }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [activeTab, confirmCurrent, removeCurrent, visible.length]);

  const highlightDetection = (listIdx) => {
    const canvas = canvasRef.current?.getCanvas();
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const img = new Image();
    img.onload = () => {
      ctx.drawImage(img, 0, 0);
      visible.forEach((det, i) => {
        const [x1, y1, x2, y2] = det.bbox;
        const sel = i === listIdx;
        ctx.fillStyle = sel ? 'rgba(0,255,0,0.25)' : 'rgba(0,0,0,0.08)';
        ctx.fillRect(x1, y1, x2-x1, y2-y1);
        ctx.strokeStyle = sel ? '#00ff00' : 'rgba(255,255,255,0.2)';
        ctx.lineWidth = sel ? 3 : 1;
        ctx.strokeRect(x1, y1, x2-x1, y2-y1);
      });
    };
    img.src = currentImage;
  };

  const handleDownload = () => {
    const canvas = canvasRef.current?.getCanvas();
    if (!canvas) return;
    const a = document.createElement('a');
    a.download = `bacteria-detection-${results.request_id}${isEnhanced ? '-enhanced' : ''}.png`;
    a.href = canvas.toDataURL('image/png');
    a.click();
  };

  const handleEnhancementApplied = (enhancedImage) => {
    setCurrentImage(enhancedImage);
    setIsEnhanced(true);
  };

  const pendingCount = visible.filter(d => !confirmedDetections.includes(d._origIdx)).length;

  return (
    <div className="results-container">
      <div className="results-header">
        <h2 className="results-title">
          Detection Results
          <span style={{ fontSize:'0.8rem', fontWeight:500, color:'#6b7280', marginLeft:'1rem' }}>
            {visible.length} detected · {confirmedDetections.length} confirmed · {removedDetections.length} removed
            {isEnhanced && <span style={{ marginLeft:'0.5rem', color:'#10b981' }}>✨ Enhanced</span>}
          </span>
        </h2>
        <div className="results-actions">
          <button className="btn btn-outline" onClick={() => setShowEnhancer(true)}>
            <Sparkles size={18}/> Enhance Image
          </button>
          <button className="btn btn-outline" onClick={handleDownload}>
            <Download size={18}/> Download
          </button>
          <button className="btn btn-primary" onClick={onReset}>
            <RotateCcw size={18}/> Analyze Another
          </button>
        </div>
      </div>

      <div className="pro-main-content">
        <div className="pro-left-panel">
          <div className="pro-tab-headers">
            <button className={`pro-tab-header ${activeTab==='visualization'?'active':''}`}
              onClick={() => setActiveTab('visualization')}>
              Full FOV Visualization
            </button>
            <button className={`pro-tab-header ${activeTab==='concentration'?'active':''}`}
              onClick={() => setActiveTab('concentration')}>
              Digital Concentration
            </button>
          </div>

          {activeTab === 'visualization' && (
            <div className="canvas-section">
              <div className="card">
                <h3 className="card-title">Detection Visualization</h3>
                
                {/* Canvas and Compact Side Panel */}
                <div className="visualization-layout">
                  {/* Canvas - Main area */}
                  <div className="canvas-wrapper">
                    <BoundingBoxCanvas
                      ref={canvasRef}
                      imageSrc={currentImage}
                      detections={visible}
                      imageDimensions={results.image_dimensions}
                      selectedDetection={selectedDetection}
                      onDetectionClick={onDetectionClick}
                    />
                  </div>
                  
                  {/* Compact Side Panel - Only tick/cross buttons */}
                  {selectedDetection && !removedDetections.includes(selectedDetection.index) && (
                    <div className="compact-side-panel">
                      <div className="compact-header">
                        <span className="compact-number">#{selectedDetection.index + 1}</span>
                        <span className="compact-conf">{(selectedDetection.confidence*100).toFixed(0)}%</span>
                      </div>
                      
                      <div className="compact-actions">
                        <button 
                          className="compact-btn confirm"
                          onClick={() => onConfirmDetection(selectedDetection.index)}
                          title="Confirm as Bacteria (C)"
                        >
                          <CheckCircle size={24}/>
                        </button>
                        <button 
                          className="compact-btn remove"
                          onClick={() => onRemoveDetection(selectedDetection.index)}
                          title="Remove False Positive (R)"
                        >
                          <XCircle size={24}/>
                        </button>
                      </div>
                      
                      <div className="compact-shortcuts">
                        <kbd>C</kbd> <span>Confirm</span>
                        <kbd>R</kbd> <span>Remove</span>
                      </div>
                    </div>
                  )}
                </div>
                
                {/* Legend */}
                <div className="legend">
                  <div className="legend-item legend-bacteria">Bacteria (Green)</div>
                  <div className="legend-item legend-spore">Spore (Orange)</div>
                  <div className="legend-item legend-selected">Selected Detection</div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'concentration' && (
            <div className="concentration-grid-view">
              <div className="batch-header">
                <h3 className="batch-title">Review All Detections</h3>
                <div className="batch-stats">
                  <div className="batch-stat confirmed"><CheckCircle size={14}/><span>{confirmedDetections.length} Confirmed</span></div>
                  <div className="batch-stat removed"><XCircle size={14}/><span>{removedDetections.length} Removed</span></div>
                  <div className="batch-stat pending"><Eye size={14}/><span>{pendingCount} Pending</span></div>
                </div>
              </div>

              {visible.length > 0 ? (
                <div className="detection-grid">
                  {visible.map((det, listIdx) => {
                    const origIdx = det._origIdx;
                    const isConfirmed = confirmedDetections.includes(origIdx);

                    return (
                      <div key={origIdx} className={`grid-cell ${isConfirmed ? 'confirmed' : ''}`}>
                        <div className="grid-img-area">
                          <DetectionCropper image={currentImage} detection={det} />
                          {isConfirmed && (
                            <div className="grid-tint confirmed">
                              <span className="grid-tint-icon"><CheckCircle size={28}/></span>
                            </div>
                          )}
                          <span className="grid-badge index">#{listIdx + 1}</span>
                          <span className="grid-badge conf">{(det.confidence*100).toFixed(0)}%</span>
                        </div>
                        <div className="grid-actions">
                          {isConfirmed ? (
                            <span className="grid-label confirmed"><CheckCircle size={11}/>Confirmed</span>
                          ) : (
                            <>
                              <button className="grid-btn tick" onClick={() => onConfirmDetection(origIdx)}>✓</button>
                              <button className="grid-btn cross" onClick={() => onRemoveDetection(origIdx)}>✕</button>
                            </>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              ) : (
                <div className="no-detections">
                  <p>All detections have been reviewed.</p>
                </div>
              )}
            </div>
          )}
        </div>

        <div className="pro-right-panel">
          <div className="thumbnail-grid-section">
            <div className="thumbnail-grid">
              <h4 className="thumbnail-title">Visible ({visible.length})</h4>
              <div className="thumbnails-scroll">
                {visible.map((det, listIdx) => {
                  const origIdx = det._origIdx;
                  const isConfirmed = confirmedDetections.includes(origIdx);
                  const isCurrent = activeTab==='concentration' && listIdx===safeIdx;
                  const isSelected = activeTab==='visualization' && selectedDetection?.index===origIdx;
                  return (
                    <div key={origIdx}
                      className={`thumbnail-item ${isCurrent||isSelected?'current':''} ${isConfirmed?'confirmed':''}`}
                      onClick={() => {
                        if (activeTab==='concentration') setCursorIndex(listIdx);
                        else { onDetectionClick(det, origIdx); highlightDetection(listIdx); }
                      }}>
                      <div className="thumbnail-number">#{listIdx+1}</div>
                      {isConfirmed && <div className="thumbnail-icon confirmed"><CheckCircle size={14}/></div>}
                      <div className="thumbnail-confidence">{(det.confidence*100).toFixed(0)}%</div>
                      <div className="thumbnail-type">{det.class_id===0?'B':'S'}</div>
                    </div>
                  );
                })}
              </div>
            </div>

            <div className="sidebar-summary">
              <h4 className="summary-title">Progress</h4>
              <div className="summary-stats-compact">
                <div className="stat-compact confirmed"><CheckCircle size={16}/><span>{confirmedDetections.length} Confirmed</span></div>
                <div className="stat-compact removed"><XCircle size={16}/><span>{removedDetections.length} Removed</span></div>
                <div className="stat-compact remaining">
                  <Eye size={16}/>
                  <span>{activeTab==='concentration' ? `${Math.max(0, visible.length - safeIdx - 1)} Remaining` : `${visible.length} Total`}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {showEnhancer && (
        <ImageEnhancerModal
          originalImage={image}
          onEnhanced={handleEnhancementApplied}
          onClose={() => setShowEnhancer(false)}
        />
      )}
      
      <style>{`
        .visualization-layout {
          display: flex;
          gap: 20px;
          margin-bottom: 20px;
          align-items: flex-start;
        }
        
        .canvas-wrapper {
          flex: 3;
          min-width: 0;
        }
        
        .compact-side-panel {
          flex: 1;
          min-width: 100px;
          max-width: 140px;
          background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%);
          border-radius: 16px;
          padding: 16px 12px;
          border: 1px solid #e2e8f0;
          box-shadow: 0 2px 8px rgba(0,0,0,0.05);
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 16px;
        }
        
        .compact-header {
          text-align: center;
          width: 100%;
        }
        
        .compact-number {
          display: block;
          font-size: 1.5rem;
          font-weight: 800;
          color: #3b82f6;
          margin-bottom: 4px;
        }
        
        .compact-conf {
          display: inline-block;
          background: #f1f5f9;
          padding: 4px 10px;
          border-radius: 20px;
          font-size: 0.75rem;
          font-weight: 700;
          color: #475569;
        }
        
        .compact-actions {
          display: flex;
          flex-direction: column;
          gap: 12px;
          width: 100%;
        }
        
        .compact-btn {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 8px;
          padding: 12px;
          border: none;
          border-radius: 12px;
          font-weight: 700;
          font-size: 0.875rem;
          cursor: pointer;
          transition: all 0.2s ease;
          width: 100%;
        }
        
        .compact-btn.confirm {
          background: linear-gradient(135deg, #10b981, #059669);
          color: white;
          box-shadow: 0 2px 8px rgba(16,185,129,0.3);
        }
        
        .compact-btn.confirm:hover {
          transform: translateY(-2px);
          box-shadow: 0 6px 16px rgba(16,185,129,0.4);
        }
        
        .compact-btn.remove {
          background: linear-gradient(135deg, #ef4444, #dc2626);
          color: white;
          box-shadow: 0 2px 8px rgba(239,68,68,0.3);
        }
        
        .compact-btn.remove:hover {
          transform: translateY(-2px);
          box-shadow: 0 6px 16px rgba(239,68,68,0.4);
        }
        
        .compact-shortcuts {
          display: flex;
          gap: 12px;
          justify-content: center;
          font-size: 0.7rem;
          color: #64748b;
          padding-top: 8px;
          border-top: 1px solid #e2e8f0;
        }
        
        .compact-shortcuts kbd {
          background: #f1f5f9;
          padding: 3px 8px;
          border-radius: 6px;
          font-weight: 700;
          font-size: 0.7rem;
          margin: 0 2px;
        }
        
        .compact-shortcuts span {
          margin-right: 8px;
        }
        
        .legend {
          display: flex;
          gap: 16px;
          flex-wrap: wrap;
          margin-top: 16px;
        }
        
        @media (max-width: 900px) {
          .visualization-layout {
            flex-direction: column;
          }
          
          .compact-side-panel {
            flex-direction: row;
            max-width: 100%;
            justify-content: space-between;
            align-items: center;
            padding: 12px 20px;
          }
          
          .compact-actions {
            flex-direction: row;
            width: auto;
          }
          
          .compact-btn {
            width: auto;
            padding: 10px 16px;
          }
          
          .compact-shortcuts {
            border-top: none;
            padding-top: 0;
          }
        }
      `}</style>
    </div>
  );
}

export default ResultsDisplay;