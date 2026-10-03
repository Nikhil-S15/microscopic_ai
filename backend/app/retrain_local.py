"""
RUN THIS LOCALLY IN VSCODE to retrain the classifier on your raw microscope images.

SETUP:
1. Create this folder structure with your raw microscope images:
   bacteria-detection-demo/
     training_data/
       Acinetobacter/    ← put raw .jpg/.png images here
       Aspergillus/
       Aureus/
       Bulkoder/
       Candida/
       Klebsiella/
       Pluribacter/

2. Run: python retrain_local.py
3. It saves bacteria_classifier_retrained.h5 → move to app/model/bacteria_cnn_6class_model.h5
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR   = "training_data"   # ← folder with 7 subfolders of raw images
OUTPUT     = "app/model/bacteria_cnn_6class_model.h5"  # overwrites old model
IMG_SIZE   = 128
BATCH_SIZE = 16
EPOCHS     = 50
# ─────────────────────────────────────────────────────────────────────────────

data_path = Path(DATA_DIR)
if not data_path.exists():
    print(f"❌ Folder '{DATA_DIR}' not found.")
    print("Create it with 7 subfolders, one per bacteria class, containing images.")
    exit(1)

classes = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
print(f"Found {len(classes)} classes: {classes}")
for c in classes:
    n = len(list((data_path / c).glob("*")))
    print(f"  {c}: {n} images")

# ── Load dataset ──────────────────────────────────────────────────────────────
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="training",
    seed=42, image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="validation",
    seed=42, image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"\nClass order (IMPORTANT — update CLASS_NAMES in main.py to match):")
for i, c in enumerate(class_names):
    print(f"  {i}: {c}")

# ── Augmentation + normalization ──────────────────────────────────────────────
augment = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.3),
    layers.RandomZoom(0.15),
    layers.RandomBrightness(0.2),
    layers.RandomContrast(0.2),
])
normalize = layers.Rescaling(1./255)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = (train_ds
    .map(lambda x, y: (normalize(augment(x, training=True)), y))
    .cache().shuffle(1000).prefetch(AUTOTUNE))
val_ds = (val_ds
    .map(lambda x, y: (normalize(x), y))
    .cache().prefetch(AUTOTUNE))

# ── Model (same architecture, trained on raw images this time) ────────────────
model = models.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Conv2D(256, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(num_classes, activation='softmax'),
])

model.compile(
    optimizer=optimizers.Adam(1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)
model.summary()

# ── Train ─────────────────────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1),
    ModelCheckpoint(OUTPUT, monitor='val_accuracy', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
]

print(f"\nTraining for up to {EPOCHS} epochs...")
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

# ── Plot ──────────────────────────────────────────────────────────────────────
plt.figure(figsize=(12, 4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Accuracy'); plt.legend()
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Loss'); plt.legend()
plt.savefig('training_results.png')
plt.show()

print(f"\n✅ Model saved to: {OUTPUT}")
print(f"   Restart uvicorn and test again.")
print(f"\n⚠️  IMPORTANT: Update CLASS_NAMES in main.py to match this order:")
for i, c in enumerate(class_names):
    print(f'   "{c}",   # index {i}')
