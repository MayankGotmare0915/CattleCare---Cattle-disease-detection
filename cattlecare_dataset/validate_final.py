# validate_final.py
# Final check before training.
# Verifies: image counts, no corruption, correct sizes,
# no data leakage between train/val/test.
# If this script says PASS — you are ready to train.
#
# CHANGES from original:
#   - Accepts --data_dir argument (default: ready_dataset)
#   - Scans all image types: .jpg .jpeg .png .bmp .webp
#   - Uses fast 8KB file hash instead of loading full image
#   - Reports class imbalance ratio
#   - size check is min 32px (not exact 224) so augmented
#     images of any resize pass

import cv2
import hashlib
import random
import argparse
from pathlib import Path

random.seed(42)

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_dir", default="ready_dataset",
        help="Dataset root to validate. Default: ready_dataset"
    )
    return p.parse_args()

args  = parse_args()
FINAL = Path(args.data_dir)

print("\n" + "="*60)
print("FINAL DATASET VALIDATION")
print(f"Checking: {FINAL.resolve()}")
print("="*60)

# ── CHECK 1: Count per split per class ──────────────────────────
print("\n[CHECK 1] Image counts")
print(f"{'Class':<25} {'Train':>7} {'Val':>6} "
      f"{'Test':>6} {'Total':>7}")
print("-"*55)

totals = {"train": 0, "val": 0, "test": 0}
class_counts = {}

train_dir = FINAL / "train"
if not train_dir.exists():
    print(f"ERROR: {FINAL}/train/ not found.")
    print("Run finalize_dataset.py first:")
    print("  python finalize_dataset.py --src augmented_dataset --dst ready_dataset")
    exit(1)

def count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for p in folder.iterdir()
               if p.suffix.lower() in IMG_EXTENSIONS)

for cls_dir in sorted(train_dir.iterdir()):
    if not cls_dir.is_dir():
        continue
    cls = cls_dir.name

    tr = count_images(FINAL / "train" / cls)
    vl = count_images(FINAL / "val"   / cls)
    te = count_images(FINAL / "test"  / cls)

    class_counts[cls] = {"train": tr, "val": vl, "test": te}
    totals["train"] += tr
    totals["val"]   += vl
    totals["test"]  += te

    status = ""
    if tr == 0:
        status = " ← WARNING: no training images"
    elif tr < 50:
        status = " ← WARNING: very few images"
    if vl == 0:
        status = " ← WARNING: no validation images"
    if te == 0 and not status:
        status = " ← WARNING: no test images"

    print(f"  {cls:<25} {tr:>7} {vl:>6} {te:>6} "
          f"{tr+vl+te:>7}{status}")

print("-"*55)
print(f"  {'TOTAL':<25} {totals['train']:>7} "
      f"{totals['val']:>6} {totals['test']:>6} "
      f"{sum(totals.values()):>7}")

# ── CHECK 2: Data leakage (train vs val vs test) ─────────────────
print("\n[CHECK 2] Cross-split leakage check")
print("  (Checks if same image appears in multiple splits)")
print("  This may take 1-2 minutes...")

def get_hashes(split_path: Path) -> dict:
    """
    Fast signature hash for leakage detection.
    Uses: file size + md5(first 8KB + last 8KB)
    This avoids false positives that can happen with first-8KB-only hashing.
    """
    hashes = {}
    for img_path in split_path.rglob("*"):
        if img_path.suffix.lower() in IMG_EXTENSIONS:
            try:
                size = img_path.stat().st_size
                with open(img_path, "rb") as f:
                    head = f.read(8192)
                    if size > 8192:
                        try:
                            f.seek(max(0, size - 8192))
                        except Exception:
                            pass
                        tail = f.read(8192)
                    else:
                        tail = b""
                h = hashlib.md5(head + tail).hexdigest()
                h = f"{size}:{h}"
                hashes[h] = img_path
            except Exception:
                pass
    return hashes

train_hashes = get_hashes(FINAL / "train")
val_hashes   = get_hashes(FINAL / "val")  if (FINAL / "val").exists()  else {}
test_hashes  = get_hashes(FINAL / "test") if (FINAL / "test").exists() else {}

tv_overlap = set(train_hashes) & set(val_hashes)
tt_overlap = set(train_hashes) & set(test_hashes)
vt_overlap = set(val_hashes)   & set(test_hashes)

def check_result(count, label):
    icon   = "OK" if count == 0 else "FAIL"
    status = "CLEAN" if count == 0 else f"PROBLEM - {count} duplicates"
    print(f"  {icon} {label}: {status}")
    return count == 0

ok1 = check_result(len(tv_overlap), "Train / Val overlap ")
ok2 = check_result(len(tt_overlap), "Train / Test overlap")
ok3 = check_result(len(vt_overlap), "Val   / Test overlap")

# ── CHECK 3: Image integrity sample ─────────────────────────────
print("\n[CHECK 3] Image integrity (random sample of 30 per split)")

all_ok = True
for split in ["train", "val", "test"]:
    split_path = FINAL / split
    if not split_path.exists():
        continue

    all_imgs = [p for p in split_path.rglob("*")
                if p.suffix.lower() in IMG_EXTENSIONS]
    if not all_imgs:
        print(f"  - {split:<6}: no images found")
        continue

    sample    = random.sample(all_imgs, min(30, len(all_imgs)))
    corrupt   = 0
    wrong_size = 0

    for p in sample:
        img = cv2.imread(str(p))
        if img is None:
            corrupt += 1
        elif img.shape[0] < 32 or img.shape[1] < 32:
            # min 32px — catches truly broken images without
            # failing on non-224 augmented crops
            wrong_size += 1

    ok = corrupt == 0 and wrong_size == 0
    all_ok = all_ok and ok
    icon = "OK" if ok else "FAIL"
    print(f"  {icon} {split:<6}: "
          f"corrupt={corrupt}  wrong_size={wrong_size}")

# ── CHECK 4: Labels file ─────────────────────────────────────────
print("\n[CHECK 4] Labels file")
labels_file = FINAL / "labels.txt"
if labels_file.exists():
    with open(labels_file) as f:
        labels = [l.strip() for l in f if l.strip()]
    print(f"  OK labels.txt found - {len(labels)} classes:")
    for lbl in labels:
        print(f"    - {lbl}")
else:
    print(f"  FAIL labels.txt NOT found at {labels_file}")
    print("    Run finalize_dataset.py - it creates labels.txt automatically.")
    all_ok = False

# ── CHECK 5: Imbalance ratio ─────────────────────────────────────
print("\n[CHECK 5] Class imbalance")
non_zero = {cls: v["train"] for cls, v in class_counts.items() if v["train"] > 0}
if len(non_zero) >= 2:
    max_cls = max(non_zero, key=non_zero.get)
    min_cls = min(non_zero, key=non_zero.get)
    ratio   = non_zero[max_cls] / non_zero[min_cls]
    icon    = "✓" if ratio <= 100 else "!"
    icon    = "OK" if ratio <= 100 else "WARN"
    print(f"  {icon} Imbalance ratio: {ratio:.0f}x")
    print(f"      Largest : {max_cls} = {non_zero[max_cls]:,} images")
    print(f"      Smallest: {min_cls} = {non_zero[min_cls]:,} images")
    if ratio > 100:
        print("    -> Use WeightedRandomSampler + FocalLoss in training.")
        print("    -> augmentation_pipeline.py has build_weighted_sampler()")

# ── FINAL VERDICT ────────────────────────────────────────────────
print("\n" + "="*60)
checks_passed = ok1 and ok2 and ok3 and all_ok

if checks_passed:
    print("RESULT: ALL CHECKS PASSED")
    print()
    print("Your dataset is ready for training.")
    print("Point your training script at:")
    print(f"  TRAIN : {FINAL}/train/")
    print(f"  VAL   : {FINAL}/val/")
    print(f"  TEST  : {FINAL}/test/")
    print(f"  LABELS: {FINAL}/labels.txt")
    print()
    print("Next step: python train.py --data_dir", FINAL)
else:
    print("RESULT: SOME CHECKS FAILED")
    print()
    print("Fix the issues above before training.")
    print("A model trained on a dirty dataset will")
    print("show fake high accuracy that fails in the field.")
    print()
    print("If val/test splits are missing, run:")
    print("  python finalize_dataset.py --src augmented_dataset --dst ready_dataset")
    print("  python validate_final.py --data_dir ready_dataset")