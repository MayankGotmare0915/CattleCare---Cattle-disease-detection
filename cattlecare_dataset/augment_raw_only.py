# augment_raw_only.py
# Reads the plan from count_and_plan.py.
# Applies augmentation ONLY to raw images (cleaned_A_raw).
# Roboflow images (cleaned_B_roboflow) are NOT touched.
# Output goes to:  augmented_from_raw/

import cv2
import numpy as np
import albumentations as A
from pathlib import Path
import random

random.seed(42)
np.random.seed(42)

# ── AUGMENTATION PIPELINES ───────────────────────────────────────
# Skin disease images: need geometric + colour variation
# because disease spots change shape/colour with lighting.
DISEASE_TRANSFORM = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=20, border_mode=cv2.BORDER_REFLECT, p=0.7),
    A.RandomBrightnessContrast(
        brightness_limit=0.30,
        contrast_limit=0.25, p=0.8),
    A.HueSaturationValue(
        hue_shift_limit=10,
        sat_shift_limit=25,
        val_shift_limit=15, p=0.6),
    A.GaussianBlur(blur_limit=(1, 3), p=0.3),
    A.ImageCompression(
        quality_lower=70,
        quality_upper=100, p=0.35),
    A.RandomShadow(
        shadow_roi=(0, 0.5, 1, 1), p=0.2),
    A.CLAHE(clip_limit=2.0,
            tile_grid_size=(4, 4), p=0.3),
])

# Healthy animals: focus on lighting/environment variation
# because healthy cows just look like cows in different settings.
HEALTHY_TRANSFORM = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT, p=0.5),
    A.RandomBrightnessContrast(
        brightness_limit=0.35,
        contrast_limit=0.30, p=0.8),
    A.HueSaturationValue(
        hue_shift_limit=15,
        sat_shift_limit=30,
        val_shift_limit=20, p=0.7),
    A.GaussianBlur(blur_limit=(1, 3), p=0.25),
    A.RandomFog(
        fog_coef_lower=0.05,
        fog_coef_upper=0.20, p=0.10),
    A.RandomSunFlare(p=0.05),
])

CLASS_TRANSFORMS = {
    "LSD"              : DISEASE_TRANSFORM,
    "Mastitis"         : DISEASE_TRANSFORM,
    "FMD"              : DISEASE_TRANSFORM,
    "BRD"              : DISEASE_TRANSFORM,
    "Mange"            : DISEASE_TRANSFORM,
    "Orf"              : DISEASE_TRANSFORM,
    "Healthy_Cow"      : HEALTHY_TRANSFORM,
}

# ── PLAN (must match count_and_plan.py output) ───────────────────
# Copy the multiplier numbers from count_and_plan.py output here.
# If a class has 0 raw images, set multiplier to 0.
PLAN = {
    "LSD"              : 1,   # 4479 raw → adjust as needed
    "Mastitis"         : 0,   # no raw images
    "FMD"              : 0,
    "BRD"              : 0,
    "Mange"            : 0,
    "Orf"              : 0,
    "Healthy_Cow"      : 2,   # 1697 raw → adjust as needed
}

# ── AUGMENT ──────────────────────────────────────────────────────
INPUT_DIR  = "cleaned_A_raw"
OUTPUT_DIR = "augmented_from_raw"

print("\n" + "="*55)
print("AUGMENTING RAW IMAGES")
print("="*55)
print(f"Input : {INPUT_DIR}")
print(f"Output: {OUTPUT_DIR}")
print()

grand_total = 0

for cls_name, multiplier in PLAN.items():
    src = Path(INPUT_DIR) / cls_name
    dst = Path(OUTPUT_DIR) / cls_name
    dst.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        print(f"  {cls_name:<25}: folder not found — skip")
        continue

    all_images = list(src.glob("*.jpg"))

    if not all_images or multiplier == 0:
        # Copy originals only (no augmentation)
        copied = 0
        for img_path in all_images:
            dest = dst / f"{img_path.stem}_orig.jpg"
            if not dest.exists():
                import shutil
                shutil.copy2(img_path, dest)
                copied += 1
        if copied > 0:
            print(f"  {cls_name:<25}: "
                  f"{copied} originals copied (no augmentation)")
        else:
            print(f"  {cls_name:<25}: 0 raw images — skip")
        grand_total += copied
        continue

    transform = CLASS_TRANSFORMS.get(
        cls_name, DISEASE_TRANSFORM)

    generated = 0

    for img_path in all_images:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        # Convert BGR → RGB (albumentations expects RGB)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Always save the ORIGINAL
        orig_dest = dst / f"{img_path.stem}_orig.jpg"
        cv2.imwrite(str(orig_dest), img_bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        generated += 1

        # Save augmented versions
        for i in range(multiplier):
            try:
                aug = transform(image=img_rgb)["image"]
                aug_bgr  = cv2.cvtColor(aug, cv2.COLOR_RGB2BGR)
                aug_dest = dst / f"{img_path.stem}_aug{i:04d}.jpg"
                cv2.imwrite(str(aug_dest), aug_bgr,
                            [cv2.IMWRITE_JPEG_QUALITY, 90])
                generated += 1
            except Exception as e:
                print(f"    Augmentation error on "
                      f"{img_path.name}: {e}")

    print(f"  {cls_name:<25}: "
          f"{len(all_images)} raw → "
          f"{generated} total (x{multiplier} aug)")
    grand_total += generated

print(f"\n  GRAND TOTAL generated: {grand_total:,} images")
print(f"\nNext step: python merge_and_split.py")