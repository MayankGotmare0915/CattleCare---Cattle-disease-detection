# merge_and_split.py
# Combines augmented raw images + Roboflow images
# into the final dataset with correct train/val/test splits.
#
# RULE (critical — never break this):
#   Roboflow images → TRAIN ONLY
#   Raw images      → split across TRAIN + VAL + TEST
#   Val and Test    → raw images only (real-world distribution)

import shutil
import random
from pathlib import Path

random.seed(42)

# ── CONFIG ───────────────────────────────────────────────────────
TRAIN_RATIO = 0.75
VAL_RATIO   = 0.15
TEST_RATIO  = 0.10    # remaining

CLASSES = [
    "LSD", "Mastitis", "FMD", "BRD", "Mange",
    "Orf", "Healthy_Cow",
]

INPUT_RAW = "augmented_from_raw"      # Step 4 output
INPUT_RBF = "cleaned_B_roboflow"      # cleaned Roboflow
OUTPUT    = "final_dataset"           # final destination

print("\n" + "="*60)
print("MERGING INTO FINAL DATASET")
print("="*60)
print(f"Raw augmented : {INPUT_RAW}")
print(f"Roboflow      : {INPUT_RBF}")
print(f"Output        : {OUTPUT}")
print()

summary = []

for cls in CLASSES:
    raw_path = Path(INPUT_RAW) / cls
    rbf_path = Path(INPUT_RBF) / cls

    # ── Gather raw-augmented images ──────────────────────────────
    raw_images = []
    if raw_path.exists():
        raw_images = list(raw_path.glob("*.jpg"))
    random.shuffle(raw_images)

    # Split raw images into train / val / test
    n       = len(raw_images)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)
    # Test gets whatever is left

    raw_splits = {
        "train" : raw_images[:n_train],
        "val"   : raw_images[n_train : n_train + n_val],
        "test"  : raw_images[n_train + n_val :],
    }

    # ── Gather Roboflow images ───────────────────────────────────
    rbf_images = []
    if rbf_path.exists():
        rbf_images = list(rbf_path.glob("*.jpg"))
    # Roboflow goes to TRAIN ONLY

    # ── Copy everything ──────────────────────────────────────────
    for split_name, images in raw_splits.items():
        out_dir = Path(OUTPUT) / split_name / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        for img in images:
            shutil.copy2(img, out_dir / img.name)

    # Roboflow → train only
    rbf_train_dir = Path(OUTPUT) / "train" / cls
    rbf_train_dir.mkdir(parents=True, exist_ok=True)
    for img in rbf_images:
        dest = rbf_train_dir / f"rbf_{img.name}"
        shutil.copy2(img, dest)

    # ── Summary row ─────────────────────────────────────────────
    final_train = len(raw_splits["train"]) + len(rbf_images)
    final_val   = len(raw_splits["val"])
    final_test  = len(raw_splits["test"])
    total       = final_train + final_val + final_test

    summary.append({
        "class" : cls,
        "train" : final_train,
        "val"   : final_val,
        "test"  : final_test,
        "total" : total,
    })

    if total > 0:
        print(f"  {cls:<25}: "
              f"train={final_train:>5}  "
              f"val={final_val:>4}  "
              f"test={final_test:>4}")
    else:
        print(f"  {cls:<25}: NO DATA — skipped")

# ── Print totals ─────────────────────────────────────────────────
print("\n" + "-"*60)
grand_train = sum(r["train"] for r in summary)
grand_val   = sum(r["val"]   for r in summary)
grand_test  = sum(r["test"]  for r in summary)
print(f"  {'TOTAL':<25}: "
      f"train={grand_train:>5}  "
      f"val={grand_val:>4}  "
      f"test={grand_test:>4}")
print(f"\n  Total images in dataset: "
      f"{grand_train + grand_val + grand_test:,}")

# ── Write classes file (needed by training script) ───────────────
train_dir = Path(OUTPUT) / "train"
actual_classes = sorted([
    d.name for d in train_dir.iterdir()
    if d.is_dir() and len(list(d.glob("*.jpg"))) > 0
])

labels_file = Path(OUTPUT) / "labels.txt"
with open(labels_file, "w") as f:
    for cls_name in actual_classes:
        f.write(cls_name + "\n")

print(f"\n  Labels file written: {labels_file}")
print(f"  Active classes ({len(actual_classes)}): "
      f"{', '.join(actual_classes)}")
print(f"\nNext step: python validate_final.py")