import React, { useEffect, useRef, forwardRef, useImperativeHandle } from 'react';

const BoundingBoxCanvas = forwardRef(({ 
  imageSrc, 
  detections, 
  imageDimensions,
  selectedDetection,
  onDetectionClick 
}, ref) => {
  const canvasRef = useRef(null);
  const imageRef = useRef(null);
  const detectionRectsRef = useRef([]);

  useImperativeHandle(ref, () => ({
    getCanvas: () => canvasRef.current,
    clearAndRedraw: () => drawDetections()
  }));

  const drawDetections = () => {
    const canvas = canvasRef.current;
    const img = imageRef.current;

    if (!canvas || !img || !img.complete) {
      return;
    }

    const ctx = canvas.getContext('2d');
    
    // Set canvas size to match image
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;

    // Draw image with subtle shadow
    ctx.save();
    ctx.shadowColor = 'rgba(0, 0, 0, 0.3)';
    ctx.shadowBlur = 10;
    ctx.drawImage(img, 0, 0);
    ctx.restore();

    detectionRectsRef.current = [];

    // Draw detections with enhanced visuals
    if (detections && detections.length > 0) {
      detections.forEach((detection, idx) => {
        const [x1, y1, x2, y2] = detection.bbox;
        const width = x2 - x1;
        const height = y2 - y1;

        // Store detection rect for click detection
        detectionRectsRef.current.push({
          x: x1, y: y1, width, height,
          detection: detection,
          index: idx
        });

        // Check if this is selected
        const isSelected = selectedDetection && 
          selectedDetection.bbox.toString() === detection.bbox.toString();

        // Use species color from detection result, fallback to blue
        let color, fillColor, glowColor;
        if (isSelected) {
          color = '#10b981';
          fillColor = 'rgba(16, 185, 129, 0.25)';
          glowColor = 'rgba(16, 185, 129, 0.6)';
        } else {
          // Use species-specific color from API response
          const hex = detection.color || '#3b82f6';
          const r   = parseInt(hex.slice(1,3),16);
          const g   = parseInt(hex.slice(3,5),16);
          const b   = parseInt(hex.slice(5,7),16);
          color     = hex;
          fillColor = `rgba(${r},${g},${b},0.15)`;
          glowColor = `rgba(${r},${g},${b},0.5)`;
        }

        // Draw filled rectangle with gradient
        const gradient = ctx.createLinearGradient(x1, y1, x1, y2);
        gradient.addColorStop(0, fillColor);
        gradient.addColorStop(1, fillColor.replace(/[\d.]+\)/, '0.05)'));
        ctx.fillStyle = gradient;
        ctx.fillRect(x1, y1, width, height);

        // Draw border with glow effect
        ctx.save();
        ctx.shadowColor = glowColor;
        ctx.shadowBlur = isSelected ? 15 : 8;
        ctx.strokeStyle = color;
        ctx.lineWidth = isSelected ? 3 : 2;
        ctx.strokeRect(x1, y1, width, height);
        ctx.restore();

        // Draw rounded corner indicators
        if (isSelected) {
          const cornerSize = 12;
          ctx.strokeStyle = color;
          ctx.lineWidth = 3;
          
          // Top-left corner
          ctx.beginPath();
          ctx.moveTo(x1, y1 + cornerSize);
          ctx.lineTo(x1, y1);
          ctx.lineTo(x1 + cornerSize, y1);
          ctx.stroke();
          
          // Top-right corner
          ctx.beginPath();
          ctx.moveTo(x2 - cornerSize, y1);
          ctx.lineTo(x2, y1);
          ctx.lineTo(x2, y1 + cornerSize);
          ctx.stroke();
          
          // Bottom-left corner
          ctx.beginPath();
          ctx.moveTo(x1, y2 - cornerSize);
          ctx.lineTo(x1, y2);
          ctx.lineTo(x1 + cornerSize, y2);
          ctx.stroke();
          
          // Bottom-right corner
          ctx.beginPath();
          ctx.moveTo(x2 - cornerSize, y2);
          ctx.lineTo(x2, y2);
          ctx.lineTo(x2, y2 - cornerSize);
          ctx.stroke();
        }

        // Draw modern label
        const label = `${detection.class_name} ${(detection.confidence * 100).toFixed(1)}%`;
        ctx.font = 'bold 13px Inter, system-ui, sans-serif';
        const textMetrics = ctx.measureText(label);
        const padding = 8;
        const labelHeight = 24;
        const labelWidth = textMetrics.width + padding * 2;

        // Position label above or below box based on space
        const labelY = y1 > labelHeight + 5 ? y1 - labelHeight - 2 : y2 + 2;

        // Draw label background with gradient
        const labelGradient = ctx.createLinearGradient(x1, labelY, x1, labelY + labelHeight);
        labelGradient.addColorStop(0, color);
        labelGradient.addColorStop(1, color.replace(/[\d.]+\)/, '0.9)'));
        
        ctx.fillStyle = labelGradient;
        ctx.shadowColor = 'rgba(0, 0, 0, 0.3)';
        ctx.shadowBlur = 8;
        
        // Rounded rectangle for label
        const radius = 6;
        ctx.beginPath();
        ctx.moveTo(x1 + radius, labelY);
        ctx.lineTo(x1 + labelWidth - radius, labelY);
        ctx.quadraticCurveTo(x1 + labelWidth, labelY, x1 + labelWidth, labelY + radius);
        ctx.lineTo(x1 + labelWidth, labelY + labelHeight - radius);
        ctx.quadraticCurveTo(x1 + labelWidth, labelY + labelHeight, x1 + labelWidth - radius, labelY + labelHeight);
        ctx.lineTo(x1 + radius, labelY + labelHeight);
        ctx.quadraticCurveTo(x1, labelY + labelHeight, x1, labelY + labelHeight - radius);
        ctx.lineTo(x1, labelY + radius);
        ctx.quadraticCurveTo(x1, labelY, x1 + radius, labelY);
        ctx.closePath();
        ctx.fill();

        // Draw label text
        ctx.shadowBlur = 0;
        ctx.fillStyle = '#ffffff';
        ctx.textBaseline = 'middle';
        ctx.fillText(label, x1 + padding, labelY + labelHeight / 2);
      });
    }
  };

  const handleCanvasClick = (e) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;

    // Check each detection rect
    for (const detectionRect of detectionRectsRef.current) {
      if (
        x >= detectionRect.x && x <= detectionRect.x + detectionRect.width &&
        y >= detectionRect.y && y <= detectionRect.y + detectionRect.height
      ) {
        onDetectionClick(detectionRect.detection, detectionRect.index);
        return;
      }
    }
  };

  useEffect(() => {
    const img = imageRef.current;
    if (img) {
      if (img.complete) {
        drawDetections();
      } else {
        img.onload = drawDetections;
      }
    }
  }, [imageSrc, detections, selectedDetection]);

  return (
    <div className="canvas-container">
      <img
        ref={imageRef}
        src={imageSrc}
        alt="Source"
        style={{ display: 'none' }}
        crossOrigin="anonymous"
      />
      <canvas
        ref={canvasRef}
        className="detection-canvas"
        onClick={handleCanvasClick}
        style={{ cursor: 'pointer' }}
      />
    </div>
  );
});

BoundingBoxCanvas.displayName = 'BoundingBoxCanvas';

export default BoundingBoxCanvas