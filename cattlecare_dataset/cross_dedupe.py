# cross_dedupe.py
# Finds images in bucket_B_roboflow that were originally sourced
# from the same photos as your raw images.
# These must be removed from the Roboflow bucket to prevent
# the same image appearing in both training and validation.

import cv2
import numpy as np
import os
from pathlib import Path

# ── HOW SIMILAR IS "TOO SIMILAR" ─────────────────────────────────
# We use perceptual hashing (pHash).
# Two images get a "distance" score from 0 to 256.
# 0   = identical
# 1-5 = nearly identical (same photo, different compression)
# 6-15 = very similar (same photo, minor augmentation applied)
# 16+  = different images
#
# We flag Roboflow images with distance <= THRESHOLD as
# "probably came from the same source photo as a raw image"

HAMMING_THRESHOLD = 10

def perceptual_hash(img, hash_size=16):
    """
    Converts an image to a 256-bit fingerprint.
    Similar images get similar fingerprints.
    """
    # Resize to small square
    small = cv2.resize(img, (hash_size, hash_size))
    # Convert to grayscale
    gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    # Compare each pixel to the mean
    mean  = gray.mean()
    bits  = (gray > mean).flatten()
    return bits

def hamming_distance(hash1, hash2):
    """Count the number of positions where bits differ."""
    return int(np.sum(hash1 != hash2))

def load_hashes(bucket_dir):
    """
    Read all images in a bucket and compute their perceptual hashes.
    Returns: {class_name: [(hash, filepath), ...]}
    """
    bucket  = Path(bucket_dir)
    db      = {}
    total   = 0

    if not bucket.exists():
        print(f"  Folder not found: {bucket_dir}")
        return db

    for class_dir in sorted(bucket.iterdir()):
        if not class_dir.is_dir():
            continue

        cls_name   = class_dir.name
        db[cls_name] = []

        for img_path in class_dir.glob("*.jpg"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h = perceptual_hash(img)
            db[cls_name].append((h, img_path))
            total += 1

    print(f"  Loaded {total} image hashes from {bucket_dir}")
    return db

def find_duplicates(db_raw, db_roboflow):
    """
    For each Roboflow image, check if any raw image is very similar.
    If yes, mark the Roboflow image for removal.
    """
    to_remove = []
    match_counts = {}

    all_classes = set(list(db_raw.keys()) +
                      list(db_roboflow.keys()))

    for cls_name in sorted(all_classes):
        raw_list = db_raw.get(cls_name, [])
        rbf_list = db_roboflow.get(cls_name, [])

        if not raw_list or not rbf_list:
            continue

        matches = 0

        for (h_rbf, rbf_path) in rbf_list:
            for (h_raw, _) in raw_list:
                dist = hamming_distance(h_rbf, h_raw)
                if dist <= HAMMING_THRESHOLD:
                    to_remove.append(rbf_path)
                    matches += 1
                    break   # found a match, no need to check more

        match_counts[cls_name] = matches
        print(f"  {cls_name:<25}: "
              f"{len(rbf_list)} Roboflow checked, "
              f"{matches} flagged as duplicates")

    return to_remove

# ── RUN ──────────────────────────────────────────────────────────
print("Loading raw image hashes...")
db_raw = load_hashes("cleaned_A_raw")

print("\nLoading Roboflow image hashes...")
db_rbf = load_hashes("cleaned_B_roboflow")

if not db_raw or not db_rbf:
    print("\nOne or both buckets are empty.")
    print("If cleaned_B_roboflow is empty, skip this step.")
    print("Next step: python count_and_plan.py")
else:
    print("\nComparing hashes (this may take a few minutes)...")
    to_remove = find_duplicates(db_raw, db_rbf)

    # Save the list to a file
    removal_list = "roboflow_to_remove.txt"
    with open(removal_list, "w") as f:
        for p in to_remove:
            f.write(str(p) + "\n")

    print(f"\nTotal Roboflow images to remove: {len(to_remove)}")
    print(f"List saved to: {removal_list}")

    if to_remove:
        print("\nRemoving flagged images...")
        removed = 0
        for img_path in to_remove:
            try:
                os.remove(img_path)
                removed += 1
            except Exception as e:
                print(f"  Could not remove {img_path}: {e}")
        print(f"Removed {removed} duplicate images.")

    print("\nNext step: python count_and_plan.py")