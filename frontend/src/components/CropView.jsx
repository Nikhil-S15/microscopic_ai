import React from 'react';

/**
 * Square, context-padded crop of one detection, rendered with CSS background
 * positioning (cheap even for hundreds of tiles). Fills its parent's width.
 */
export default function CropView({ src, width, height, bbox, color, context = 2.6, dimmed = false }) {
  const [x1, y1, x2, y2] = bbox;
  const side = Math.min(Math.max(Math.max(x2 - x1, y2 - y1) * context, 96), width, height);
  const cx = (x1 + x2) / 2;
  const cy = (y1 + y2) / 2;
  const sx = Math.min(Math.max(cx - side / 2, 0), width - side);
  const sy = Math.min(Math.max(cy - side / 2, 0), height - side);
  const posX = width === side ? 0 : (sx / (width - side)) * 100;
  const posY = height === side ? 0 : (sy / (height - side)) * 100;

  return (
    <span
      className={`crop ${dimmed ? 'dimmed' : ''}`}
      style={{
        backgroundImage: `url(${src})`,
        backgroundSize: `${(width / side) * 100}% ${(height / side) * 100}%`,
        backgroundPosition: `${posX}% ${posY}%`,
      }}
    >
      <span
        className="crop-box"
        style={{
          left: `${((x1 - sx) / side) * 100}%`,
          top: `${((y1 - sy) / side) * 100}%`,
          width: `${((x2 - x1) / side) * 100}%`,
          height: `${((y2 - y1) / side) * 100}%`,
          borderColor: color,
        }}
      />
    </span>
  );
}
