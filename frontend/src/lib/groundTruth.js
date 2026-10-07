/**
 * Ground-truth label files, matched to samples by file name (sample "12.jpg" ↔ "12.txt").
 *
 * One object per line, YOLO style: `class x_center y_center width height`, or a
 * polygon `class x1 y1 x2 y2 x3 y3 …`. Coordinates normalised to 0–1 (pixel
 * coordinates are accepted too). The class column is ignored: every labelled
 * object counts as a detection target.
 */

export const fileStem = (name) => name.replace(/\.[^./\\]+$/, '').toLowerCase();

/** Parse a label file into pixel boxes [x1, y1, x2, y2]. Throws on malformed lines. */
export function parseLabels(text, width, height) {
  const boxes = [];
  text.split(/\r?\n/).forEach((line, i) => {
    const parts = line.trim().split(/[\s,]+/).filter(Boolean);
    if (!parts.length) return;
    const v = parts.slice(1).map(Number);
    if (v.some((n) => !Number.isFinite(n))) throw new Error(`line ${i + 1}: not a number`);
    let xs, ys;
    if (v.length === 4) {
      const [cx, cy, w, h] = v;
      xs = [cx - w / 2, cx + w / 2];
      ys = [cy - h / 2, cy + h / 2];
    } else if (v.length >= 6 && v.length % 2 === 0) {
      xs = v.filter((_, k) => k % 2 === 0);
      ys = v.filter((_, k) => k % 2 === 1);
    } else {
      throw new Error(`line ${i + 1}: expected 4 box values or a polygon`);
    }
    const normalised = Math.max(...xs, ...ys) <= 1.0001;
    const sx = normalised ? width : 1;
    const sy = normalised ? height : 1;
    const box = [
      Math.max(0, Math.min(...xs) * sx), Math.max(0, Math.min(...ys) * sy),
      Math.min(width, Math.max(...xs) * sx), Math.min(height, Math.max(...ys) * sy),
    ];
    if (box[2] > box[0] && box[3] > box[1]) boxes.push(box);
  });
  return boxes;
}

/** Read dropped files into { stem: { name, text } }. */
export async function readLabelFiles(files) {
  const out = {};
  await Promise.all(files.map(async (f) => { out[fileStem(f.name)] = { name: f.name, text: await f.text() }; }));
  return out;
}

/**
 * Pair analysed samples with their labels.
 * Returns { images: [{ item, boxes }], unlabelled: [item], errors: [{ name, error }], unused: [labelName] }.
 * With `emptyIfMissing`, samples without a label file count as having no objects.
 */
export function pairLabels(items, labels, emptyIfMissing) {
  const images = [];
  const unlabelled = [];
  const errors = [];
  const used = new Set();
  for (const item of items) {
    const stem = fileStem(item.file.name);
    const lf = labels[stem];
    if (!lf) {
      if (emptyIfMissing) images.push({ item, boxes: [] });
      else unlabelled.push(item);
      continue;
    }
    used.add(stem);
    try {
      images.push({ item, boxes: parseLabels(lf.text, item.result.image.width, item.result.image.height) });
    } catch (e) {
      errors.push({ name: lf.name, error: e.message });
    }
  }
  const unused = Object.entries(labels).filter(([s]) => !used.has(s)).map(([, lf]) => lf.name);
  return { images, unlabelled, errors, unused };
}
