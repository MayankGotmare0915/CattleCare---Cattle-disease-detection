# count_and_plan.py
# Counts images per class in both cleaned buckets.
# Calculates exactly how many augmented versions per raw
# image are needed to reach the target count.
# Prints a clear plan before you do any augmentation.

from pathlib import Path

# ── CONFIG ───────────────────────────────────────────────────────
# Minimum number of images per class in the TRAINING set.
# Adjust this based on what you actually have.
# Realistic targets given your current data:
#   If you have < 500 raw  → target 1500 (augment heavily)
#   If you have 500-2000   → target 2000
#   If you have 2000+      → target 3000

TARGET_PER_CLASS = 2000

# Maximum multiplier — don't augment more than this many times
# Augmenting more than 10x on very small datasets (< 50 images)
# causes the model to memorise augmentation patterns
MAX_MULTIPLIER = 12

CLASSES = [
    "LSD", "Mastitis", "FMD", "BRD", "Mange",
    "Healthy_Cow"
]

def count_images(directory, cls_name):
    path = Path(directory) / cls_name
    if not path.exists():
        return 0
    return len(list(path.glob("*.jpg")))

# ── COUNT ────────────────────────────────────────────────────────
print("\n" + "="*75)
print("DATASET COUNT AND AUGMENTATION PLAN")
print("="*75)

header = (f"{'Class':<25} {'Raw':>6} {'Roboflow':>9} "
          f"{'Total':>7} {'Mult':>6} {'After Aug':>10} {'Status':>10}")
print(header)
print("-"*75)

plan = {}
warnings = []

for cls in CLASSES:
    raw_count = count_images("cleaned_A_raw", cls)
    rbf_count = count_images("cleaned_B_roboflow", cls)
    total_now = raw_count + rbf_count

    # Calculate needed images from raw augmentation
    # Roboflow images go directly to train (no augmentation needed)
    needed_from_aug = max(TARGET_PER_CLASS - rbf_count, 0)

    if raw_count == 0:
        multiplier = 0
        after_aug  = rbf_count
        status     = "NO DATA"
        warnings.append(cls)
    else:
        # How many augmented versions per original image
        # (raw_count originals + raw_count * mult augmented)
        # total_from_raw = raw_count * (1 + mult)
        # We want total_from_raw >= needed_from_aug
        mult_needed = max(0, needed_from_aug // raw_count - 1)
        multiplier  = min(mult_needed, MAX_MULTIPLIER)
        after_aug   = rbf_count + raw_count * (1 + multiplier)
        status      = ("OK" if after_aug >= TARGET_PER_CLASS
                       else "LOW")

    plan[cls] = {
        "raw"        : raw_count,
        "roboflow"   : rbf_count,
        "multiplier" : multiplier,
        "after_aug"  : after_aug,
    }

    print(f"{cls:<25} {raw_count:>6} {rbf_count:>9} "
          f"{total_now:>7} {multiplier:>5}x "
          f"{after_aug:>10} {status:>10}")

print("-"*75)
total_train_est = sum(
    p["after_aug"] for p in plan.values())
print(f"\nEstimated total training images: {total_train_est:,}")
print(f"Target per class               : {TARGET_PER_CLASS:,}")

# ── WARNINGS ─────────────────────────────────────────────────────
if warnings:
    print(f"\n{'='*75}")
    print("CLASSES WITH NO DATA (cannot train on these):")
    for cls in warnings:
        print(f"  ✗ {cls}")
    print("\nOptions for missing classes:")
    print("  1. Re-check Roboflow downloads — run diagnose.py")
    print("  2. Download from Kaggle (see download_kaggle.py)")
    print("  3. Scrape CFSPH Iowa State disease images")
    print("  4. For hackathon MVP: train on available "
          "classes only")
    print(f"\n  RECOMMENDATION: If < 5 classes have data,")
    print(f"  train a 2-class model first (LSD + Healthy).")
    print(f"  Expand as you get more datasets.")

print(f"\nNext step: python augment_raw_only.py")