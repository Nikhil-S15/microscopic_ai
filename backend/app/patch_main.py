"""
Run this from your backend folder:
cd /Users/nikhils/Documents/development/Bacteria_detection/bacteria-detection-demo/backend
python patch_main.py
"""

content = open('main.py').read()

# Find exact current preprocessing block
import re
lines = content.split('\n')
for i, l in enumerate(lines):
    if 'gray    = cv2.cvtColor' in l or 'gray     = cv2.cvtColor' in l:
        print(f"Found gray conversion at line {i+1}")
        for j in range(max(0,i-2), min(len(lines), i+12)):
            print(f"{j+1}: {lines[j]}")
        break

# The new preprocessing block — smart auto-detection
NEW_PREPROCESS = '''        # ── Smart auto-preprocessing ─────────────────────────────────────
        # Training format: BLACK background (~99.3%), WHITE bacteria (~0.7%)
        # Raw microscope images can be either dark or bright background
        # We auto-detect and normalize to match training format
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Detect image type by mean brightness
        mean_val = gray.mean()
        logger.info(f"Raw image gray mean: {mean_val:.2f}")

        if mean_val > 127:
            # BRIGHT background image → invert to get dark background
            gray = cv2.bitwise_not(gray)
            logger.info("Bright image detected → inverted")

        # Now apply CLAHE to enhance bacteria visibility
        clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray    = clahe.apply(gray)

        # Resize to 128x128 and normalize
        resized = cv2.resize(gray, (128, 128))
        arr     = resized.astype(np.float32) / 255.0
        arr     = arr.reshape(1, 128, 128, 1)
        logger.info(f"Classifier input: mean={arr.mean():.4f}")'''

# Try to find and replace the current preprocessing
found = False

# Pattern 1: current no-inversion version
old1 = '''        # Training images: dark bg, white bacteria, mean~0.003 — NO inversion needed
        gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (128, 128))
        arr     = resized.astype(np.float32) / 255.0
        arr     = arr.reshape(1, 128, 128, 1)
        logger.info(f"Classifier input: mean={arr.mean():.4f} (should be ~0.003)")'''

# Pattern 2: with inversion
old2 = '''        # Training images are already dark (black bg, white bacteria, mean~0.003)
        # DO NOT invert — just grayscale → resize → normalize
        gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (128, 128))
        arr     = resized.astype(np.float32) / 255.0
        arr     = arr.reshape(1, 128, 128, 1)
        logger.info(f"Classifier input: mean={arr.mean():.3f} (should be ~0.003)")'''

for old in [old1, old2]:
    if old in content:
        content = content.replace(old, NEW_PREPROCESS)
        found = True
        print("\n✅ Preprocessing block replaced!")
        break

if not found:
    print("\n❌ Could not find preprocessing block automatically.")
    print("Add this manually in main.py inside classify_bacteria(), replacing the gray/resize lines:\n")
    print(NEW_PREPROCESS)
else:
    open('main.py', 'w').write(content)
    print("✅ main.py updated successfully!")
    print("Uvicorn will auto-reload.")
