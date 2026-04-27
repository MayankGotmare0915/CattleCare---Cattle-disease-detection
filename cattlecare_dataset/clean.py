# clean.py
# Validates, deduplicates, and resizes all images in both buckets.
# Run this on bucket_A_raw and bucket_B_roboflow.

import cv2
import hashlib
import os
from pathlib import Path

# ── CONFIG ───────────────────────────────────────────────────────
TARGET_SIZE = (224, 224)   # MobileNetV3 input size
MIN_DIM     = 80           # reject images smaller than this
JPEG_QUALITY = 95

# ── WHAT TO CLEAN ────────────────────────────────────────────────
# Input  → Output
BUCKETS = [
    ("bucket_A_raw",      "cleaned_A_raw"),
    ("bucket_B_roboflow", "cleaned_B_roboflow"),
]

def clean_one_bucket(input_dir, output_dir):
    print(f"\n{'='*55}")
    print(f"Cleaning: {input_dir}  →  {output_dir}")
    print(f"{'='*55}")

    stats = {
        "kept"      : 0,
        "corrupt"   : 0,
        "too_small" : 0,
        "duplicate" : 0,
        "skipped"   : 0,
    }

    input_path  = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        print(f"  Input folder not found. Skipping.")
        return stats

    # Process class by class
    class_dirs = [d for d in input_path.iterdir() if d.is_dir()]

    if not class_dirs:
        print(f"  No class subfolders found in {input_dir}")
        return stats

    for class_dir in sorted(class_dirs):
        cls_name = class_dir.name
        out_cls  = output_path / cls_name
        out_cls.mkdir(parents=True, exist_ok=True)

        # Collect all image files recursively
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png',
                    '*.JPG', '*.JPEG', '*.PNG', '*.bmp']:
            image_files.extend(class_dir.rglob(ext))

        seen_hashes = set()
        class_kept  = 0

        for img_path in image_files:
            try:
                # Read image
                img = cv2.imread(str(img_path))

                # Check 1: Can we read it at all?
                if img is None:
                    stats["corrupt"] += 1
                    continue

                # Check 2: Is it big enough?
                h, w = img.shape[:2]
                if h < MIN_DIM or w < MIN_DIM:
                    stats["too_small"] += 1
                    continue

                # Check 3: Is it a duplicate?
                # We hash the raw pixels BEFORE resizing
                raw_hash = hashlib.md5(img.tobytes()).hexdigest()
                if raw_hash in seen_hashes:
                    stats["duplicate"] += 1
                    continue
                seen_hashes.add(raw_hash)

                # Resize to target size
                img_resized = cv2.resize(
                    img, TARGET_SIZE,
                    interpolation=cv2.INTER_LANCZOS4)

                # Save with hash as filename (guarantees unique names)
                out_name = f"{raw_hash}.jpg"
                out_file = out_cls / out_name
                cv2.imwrite(
                    str(out_file), img_resized,
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

                stats["kept"] += 1
                class_kept += 1

            except Exception as e:
                stats["corrupt"] += 1

        print(f"  {cls_name:<25}: {class_kept} images kept")

    print(f"\n  Summary for {input_dir}:")
    print(f"    Kept      : {stats['kept']}")
    print(f"    Corrupt   : {stats['corrupt']}")
    print(f"    Too small : {stats['too_small']}")
    print(f"    Duplicates: {stats['duplicate']}")
    return stats

# ── RUN ──────────────────────────────────────────────────────────
all_stats = {}
for input_dir, output_dir in BUCKETS:
    all_stats[input_dir] = clean_one_bucket(input_dir, output_dir)

print(f"\n{'='*55}")
print("ALL CLEANING DONE")
grand_total = sum(s["kept"] for s in all_stats.values())
print(f"Grand total images kept: {grand_total}")
print(f"Next step: run  python cross_dedupe.py")