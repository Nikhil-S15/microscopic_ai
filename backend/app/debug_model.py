"""
Run from backend folder:
cd bacteria-detection-demo/backend
source /Users/nikhils/Documents/development/Bacteria_detection/venv/bin/activate
python app/debug_model.py
"""
import tensorflow as tf
import numpy as np
import os

# Try to find the exact model file
model_dir = "app/model"
print("Files in model directory:")
for f in os.listdir(model_dir):
    size = os.path.getsize(os.path.join(model_dir, f))
    print(f"  {f}  ({size/1024/1024:.1f} MB)")

MODEL_PATH = "app/model/bacteria_cnn_6class_model (1).h5"

CLASS_NAMES = [
    "Acenetobacter_Preprocessed",
    "Aspergillus_Preprocessed",
    "Aureus_Preprocessed",
    "Bulkoder_Preprocessed",
    "Candida_Preprocessed",
    "Klebsiella_Preprocessed",
    "Pluribacter_Preprocessed",
]

print(f"\nLoading: {MODEL_PATH}")

# Keras 3 with TF backend needs this for legacy h5
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

try:
    # Method 1: tf.keras with legacy flag
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ Loaded with Method 1 (load_model compile=False)")
except Exception as e1:
    print(f"Method 1 failed: {e1}")
    try:
        # Method 2: h5py direct
        import h5py
        import keras
        with h5py.File(MODEL_PATH, 'r') as f:
            print("H5 keys:", list(f.keys()))
        model = keras.saving.load_model(MODEL_PATH, compile=False)
        print("✅ Loaded with Method 2 (keras.saving)")
    except Exception as e2:
        print(f"Method 2 failed: {e2}")
        try:
            # Method 3: use tf.compat.v1
            model = tf.compat.v2.keras.models.load_model(MODEL_PATH, compile=False)
            print("✅ Loaded with Method 3 (tf.compat.v2.keras)")
        except Exception as e3:
            print(f"Method 3 failed: {e3}")
            print("\n❌ All methods failed. Model file may be corrupted.")
            print("Please re-export from Colab using: model.save('model.keras')")
            exit(1)

print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)

def predict(arr, label):
    preds = model(arr, training=False).numpy()[0]
    idx = np.argmax(preds)
    print(f"  [{label}]")
    for i, (n, p) in enumerate(zip(CLASS_NAMES, preds)):
        marker = " <<<" if i == idx else ""
        print(f"    {n}: {p*100:.2f}%{marker}")
    print()

black  = np.zeros((1,128,128,1), dtype=np.float32)
white  = np.ones((1,128,128,1),  dtype=np.float32)
np.random.seed(42)
noise  = np.random.rand(1,128,128,1).astype(np.float32)
sparse = np.zeros((1,128,128,1), dtype=np.float32)
for _ in range(50):
    x,y = np.random.randint(0,128), np.random.randint(0,128)
    sparse[0,x,y,0] = 1.0

predict(black,  "Pure BLACK")
predict(white,  "Pure WHITE")
predict(noise,  "Random NOISE")
predict(sparse, "Sparse dots on black (like training)")