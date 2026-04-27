# train.py
# Run from: C:\Projects\CattleCare\cattlecare_dataset\
# Command : python train.py

import os
import gc
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight

# ── SUPPRESS TF INFO LOGS (reduces noise) ────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ── CONFIG ───────────────────────────────────────────────────────
DATASET_DIR  = Path("C:/Projects/CattleCare/cattlecare_dataset/ready_dataset")
MODEL_DIR    = Path("C:/Projects/CattleCare/models")
LOG_DIR      = Path("C:/Projects/CattleCare/logs")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True,   exist_ok=True)

IMG_SIZE    = 224
BATCH_SIZE  = 4     # reduced from 8 to 4 to prevent laptop crashes
EPOCHS_P1   = 8     # Phase 1: frozen base
EPOCHS_P2   = 60    # Phase 2: fine-tuning (increased to allow reaching 0.92+ accuracy)
SEED        = 42
RUN_ID      = datetime.now().strftime("%Y%m%d_%H%M")

print("="*55)
print("CATTLECARE AI — MODEL TRAINING")
print("="*55)
print(f"Run ID     : {RUN_ID}")
print(f"Dataset    : {DATASET_DIR}")
print(f"Image size : {IMG_SIZE}x{IMG_SIZE}")
print(f"Batch size : {BATCH_SIZE}")

# ── LOAD DATASETS ────────────────────────────────────────────────
print("\nLoading datasets...")

# Read class names from labels.txt
labels_file = DATASET_DIR / "labels.txt"
if labels_file.exists():
    with open(labels_file) as f:
        CLASS_NAMES = [l.strip() for l in f if l.strip()]
else:
    # Auto-detect from train folder
    CLASS_NAMES = sorted([
        d.name for d in (DATASET_DIR/"train").iterdir()
        if d.is_dir() and
        len(list(d.glob("*.jpg"))) > 0
    ])

NUM_CLASSES = len(CLASS_NAMES)
print(f"Classes ({NUM_CLASSES}): {CLASS_NAMES}")

def make_dataset(split, shuffle=True):
    ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR / split,
        labels        = "inferred",
        label_mode    = "categorical",
        class_names   = CLASS_NAMES,
        image_size    = (IMG_SIZE, IMG_SIZE),
        batch_size    = BATCH_SIZE,
        shuffle       = shuffle,
        seed          = SEED,
    )
    # Prefetch for faster loading (removed .cache() to prevent RAM overflow)
    AUTOTUNE = tf.data.AUTOTUNE
    if shuffle:
        ds = ds.shuffle(1000, seed=SEED).prefetch(AUTOTUNE)
    else:
        ds = ds.prefetch(AUTOTUNE)
    return ds

train_ds = make_dataset("train", shuffle=True)
val_ds   = make_dataset("val",   shuffle=False)
test_ds  = make_dataset("test",  shuffle=False)

# Count images per class for class weights
train_counts = []
for cls in CLASS_NAMES:
    cls_path = DATASET_DIR / "train" / cls
    count    = len(list(cls_path.glob("*.jpg")))
    train_counts.append(count)
    print(f"  train/{cls:<25}: {count}")

# ── CLASS WEIGHTS ────────────────────────────────────────────────
# Critical for imbalanced datasets.
# LSD has 8000 images, Orf has 6 — without weights,
# the model ignores rare classes completely.
labels_for_weight = []
for cls_idx, count in enumerate(train_counts):
    labels_for_weight.extend([cls_idx] * count)

labels_array = np.array(labels_for_weight)
class_weights_array = compute_class_weight(
    class_weight = "balanced",
    classes      = np.unique(labels_array),
    y            = labels_array
)
class_weight_dict = {
    i: float(w) for i, w in enumerate(class_weights_array)
}
print(f"\nClass weights: {class_weight_dict}")

# ── MODEL DEFINITION ─────────────────────────────────────────────
def build_model(num_classes, dropout_rate=0.4):
    """
    MobileNetV3-Small pretrained on ImageNet.
    Small = optimized for mobile (< 15MB after quantization).
    Large = higher accuracy but 2x bigger model.
    Start with Small. Switch to Large only if accuracy < 80%.
    """
    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape       = (IMG_SIZE, IMG_SIZE, 3),
        include_top       = False,
        weights           = "imagenet",
        include_preprocessing = True,  # handles pixel scaling
    )
    base_model.trainable = False  # freeze for Phase 1

    inputs  = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x       = base_model(inputs, training=False)
    x       = tf.keras.layers.GlobalAveragePooling2D()(x)
    x       = tf.keras.layers.BatchNormalization()(x)
    x       = tf.keras.layers.Dropout(dropout_rate)(x)
    x       = tf.keras.layers.Dense(256, activation="relu")(x)
    x       = tf.keras.layers.Dropout(dropout_rate / 2)(x)
    outputs = tf.keras.layers.Dense(
                  num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    return model, base_model

checkpoint_path_p2 = MODEL_DIR / "best_phase2.keras"
checkpoint_path_p1 = MODEL_DIR / "best_phase1.keras"
skip_phase1 = False
already_unfrozen = False

if checkpoint_path_p2.exists():
    print(f"\nResuming from Phase 2 checkpoint: {checkpoint_path_p2}")
    model = tf.keras.models.load_model(checkpoint_path_p2.as_posix())
    skip_phase1 = True
    already_unfrozen = True
    print(f"Phase 2 Model loaded successfully. Ready to push for 92% accuracy!")
elif checkpoint_path_p1.exists():
    print(f"\nResuming from Phase 1 checkpoint: {checkpoint_path_p1}")
    model = tf.keras.models.load_model(checkpoint_path_p1.as_posix())
    skip_phase1 = True
    print(f"Phase 1 Model loaded successfully. Skipping Phase 1.")
else:
    model, base_model = build_model(NUM_CLASSES)
    print(f"\nModel built.")

# Find the base_model inside the loaded model to unfreeze it later
base_model = None
for layer in model.layers:
    if "mobilenet" in layer.name.lower() or isinstance(layer, tf.keras.Model):
        base_model = layer
        break

print(f"Total params    : {model.count_params():,}")
print(f"Trainable params: "
      f"{sum([np.prod(v.shape) for v in model.trainable_variables]):,}")

# ── CALLBACKS ────────────────────────────────────────────────────
def make_callbacks(phase_name):
    return [
        # Save the best model (based on val_accuracy)
        tf.keras.callbacks.ModelCheckpoint(
            filepath   = str(MODEL_DIR / f"best_{phase_name}.keras"),
            monitor    = "val_accuracy",
            save_best_only = True,
            verbose    = 1,
        ),

        # Stop early if no improvement for 15 epochs
        tf.keras.callbacks.EarlyStopping(
            monitor              = "val_loss",
            patience             = 15,
            restore_best_weights = True,
            verbose              = 1,
        ),

        # Reduce LR when stuck
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor   = "val_loss",
            factor    = 0.5,
            patience  = 5,
            min_lr    = 1e-8,
            verbose   = 1,
        ),

        # Log training history to CSV (open in Excel)
        tf.keras.callbacks.CSVLogger(
            filename = str(LOG_DIR / f"history_{phase_name}_{RUN_ID}.csv"),
            append   = False,
        ),

        # Print progress every 5 epochs
        tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: print(
                f"\n  Epoch {epoch+1:02d} | "
                f"loss={logs['loss']:.4f} | "
                f"acc={logs['accuracy']:.4f} | "
                f"val_loss={logs['val_loss']:.4f} | "
                f"val_acc={logs['val_accuracy']:.4f}"
            ) if (epoch + 1) % 5 == 0 else None
        ),
        
        # Clear memory after each epoch to prevent crashing
        tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: gc.collect()
        ),
    ]

# ── PHASE 1: TRAIN CLASSIFIER HEAD ONLY ─────────────────────────
if not skip_phase1:
    print("\n" + "="*55)
    print("PHASE 1: Training classifier head (base frozen)")
    print(f"  Epochs: {EPOCHS_P1}")
    print(f"  LR    : 1e-3")
    print("="*55)

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss      = "categorical_crossentropy",
        metrics   = [
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ]
    )

    history_p1 = model.fit(
        train_ds,
        validation_data = val_ds,
        epochs          = EPOCHS_P1,
        class_weight    = class_weight_dict,
        callbacks       = make_callbacks("phase1"),
        verbose         = 1,
    )

    p1_val_acc = max(history_p1.history["val_accuracy"])
    print(f"\nPhase 1 best val accuracy: {p1_val_acc:.4f}")
else:
    print("\n" + "="*55)
    print("PHASE 1: SKIPPED (Loaded from checkpoint)")
    print("="*55)
    p1_val_acc = 0.781  # Approximate accuracy from loaded checkpoint

# ── PHASE 2: FINE-TUNE TOP LAYERS ────────────────────────────────
print("\n" + "="*55)
print("PHASE 2: Fine-tuning (unfreezing top 30 layers)")
print(f"  Epochs: {EPOCHS_P2}")
print(f"  LR    : 1e-4  (10x lower than Phase 1)")
print("="*55)

# Unfreeze top 30 layers of base if not already unfrozen
if not already_unfrozen:
    if base_model is None:
        raise ValueError("Could not find base_model in the loaded model layers. Cannot fine-tune.")

    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
else:
    print("Model is already fine-tuned/unfrozen. Proceeding directly to continued training.")

trainable_count = sum(
    np.prod(v.shape)
    for v in model.trainable_variables)
print(f"Trainable params now: {trainable_count:,}")

# Must recompile after changing trainability
model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss      = "categorical_crossentropy",
    metrics   = [
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ]
)

history_p2 = model.fit(
    train_ds,
    validation_data = val_ds,
    epochs          = EPOCHS_P2,
    class_weight    = class_weight_dict,
    callbacks       = make_callbacks("phase2"),
    verbose         = 1,
)

# ── SAVE FINAL MODEL ─────────────────────────────────────────────
final_model_path = MODEL_DIR / f"cattlecare_final_{RUN_ID}.keras"
model.save(str(final_model_path))

# Save class names alongside model
with open(MODEL_DIR / "labels.txt", "w") as f:
    for name in CLASS_NAMES:
        f.write(name + "\n")

# Save training summary
summary = {
    "run_id"          : RUN_ID,
    "classes"         : CLASS_NAMES,
    "num_classes"     : NUM_CLASSES,
    "phase1_best_acc" : float(p1_val_acc),
    "phase2_best_acc" : float(max(history_p2.history["val_accuracy"])),
    "model_path"      : str(final_model_path),
}
with open(MODEL_DIR / f"summary_{RUN_ID}.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nTraining complete.")
print(f"Best val accuracy: {summary['phase2_best_acc']:.4f}")
print(f"Model saved      : {final_model_path}")
print(f"\nNext step: python evaluate.py")