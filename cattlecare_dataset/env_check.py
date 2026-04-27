# env_check.py
# Run this FIRST. It tells you exactly what you have
# and what training speed to expect.

import sys
import platform

print("="*55)
print("ENVIRONMENT REPORT")
print("="*55)

# Python version
print(f"\nPython   : {sys.version}")
print(f"Platform : {platform.platform()}")

# TensorFlow
try:
    import tensorflow as tf
    print(f"TensorFlow: {tf.__version__}")

    # GPU check
    gpus = tf.config.list_physical_devices('GPU')
    cpus = tf.config.list_physical_devices('CPU')

    print(f"\nCPU devices : {len(cpus)}")
    print(f"GPU devices : {len(gpus)}")

    if gpus:
        for gpu in gpus:
            print(f"  GPU found: {gpu}")
        print("\nSTATUS: GPU training available ✓")
    else:
        print("\nSTATUS: No GPU detected.")
        print("  -> Running on CPU only.")
        print("  -> Expected training time: 2-4 hours per epoch")
        print("     on a modern CPU with 6,000 images.")
        print("  -> This is acceptable for your dataset size.")

except ImportError:
    print("TensorFlow NOT installed.")
    print("Run: pip install tensorflow==2.10.0")

# Check dataset exists
import os
from pathlib import Path

DATASET = Path("C:/Projects/CattleCare/cattlecare_dataset/ready_dataset")
print(f"\n{'='*55}")
print("DATASET CHECK")
print(f"{'='*55}")

if DATASET.exists():
    for split in ["train", "val", "test"]:
        split_path = DATASET / split
        if split_path.exists():
            classes = [d for d in split_path.iterdir()
                       if d.is_dir()]
            total   = sum(
                len(list(c.glob("*.jpg"))) for c in classes)
            print(f"  {split:<6}: {len(classes)} classes, "
                  f"{total} images")
        else:
            print(f"  {split:<6}: NOT FOUND")
else:
    print(f"  Dataset folder not found: {DATASET}")
    print("  Run the pipeline scripts first.")

print(f"\n{'='*55}")
print("Ready to proceed with training plan above.")