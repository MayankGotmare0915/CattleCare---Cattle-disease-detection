"""
=============================================================
 CATTLE DISEASE DATASET — AUGMENTATION PIPELINE
=============================================================
 Strategies implemented:
   1. Geometric transforms      (all classes, 2–3× multiplier)
   2. Photometric transforms    (all classes, 2× multiplier)
   3. MixUp + CutMix            (scarce classes, 1.5×, online)
   4. Synthetic image intake    (BRD, Orf — drop folder here)
   5. Weighted sampler helper   (for DataLoader)

 Expected input structure:
   final_dataset/
     train/  val/  test/
       BRD/  FMD/  Healthy_Cow/  LSD/  Mastitis/  Orf/
         *.jpg / *.png

 Usage:
   python augmentation_pipeline.py --data_dir final_dataset \
          --output_dir augmented_dataset --seed 42

 Requirements:
   pip install albumentations opencv-python-headless tqdm torch torchvision
=============================================================
"""

import os
import cv2
import random
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

CLASSES = ["BRD", "FMD", "Healthy_Cow", "LSD", "Mastitis", "Orf"]
SPLITS  = ["train", "val", "test"]

# Multipliers per strategy (train split only; val/test untouched)
GEO_MULTIPLIER    = 2   # geometric  → ×2 extra copies per image
PHOTO_MULTIPLIER  = 2   # photometric → ×2 extra copies per image

# Classes that get extra MixUp/CutMix offline copies
SCARCE_THRESHOLD  = 500          # classes with fewer train images than this
MIXUP_MULTIPLIER  = 2            # extra copies via MixUp for scarce classes

# Synthetic intake folder (drop vet-verified images here)
# Structure: synthetic_intake/BRD/*.jpg  synthetic_intake/Orf/*.jpg
SYNTHETIC_DIR     = "synthetic_intake"

IMG_EXTENSIONS    = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ─────────────────────────────────────────────────────────────
# STRATEGY 1 — GEOMETRIC TRANSFORMS
# ─────────────────────────────────────────────────────────────

def _make_random_resized_crop():
    """
    RandomResizedCrop API differs between albumentations versions:
      v1.x  → height=, width=
      v2.x  → size=(h, w)
    This helper detects the installed version and calls correctly.
    """
    import albumentations as _A
    import inspect
    sig = inspect.signature(_A.RandomResizedCrop.__init__)
    if "size" in sig.parameters:
        # albumentations >= 2.0
        return _A.RandomResizedCrop(
            size=(224, 224), scale=(0.80, 1.00), ratio=(0.9, 1.1), p=0.6
        )
    else:
        # albumentations < 2.0
        return _A.RandomResizedCrop(
            height=224, width=224, scale=(0.80, 1.00), ratio=(0.9, 1.1), p=0.6
        )

geo_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.7),
    _make_random_resized_crop(),
    A.Resize(224, 224),   # ensure consistent output size
])

# ─────────────────────────────────────────────────────────────
# STRATEGY 2 — PHOTOMETRIC TRANSFORMS
# ─────────────────────────────────────────────────────────────

def _make_image_compression(p=0.4):
    import inspect
    sig = inspect.signature(A.ImageCompression.__init__)
    if "quality" in sig.parameters:
        return A.ImageCompression(quality=(70, 95), p=p)
    return A.ImageCompression(quality_lower=70, quality_upper=95, p=p)

def _make_gauss_noise():
    import inspect
    sig = inspect.signature(A.GaussNoise.__init__)
    if "std_range" in sig.parameters:
        return A.GaussNoise(std_range=(0.01, 0.05), p=0.3)
    return A.GaussNoise(var_limit=(5.0, 25.0), p=0.3)

photo_transform = A.Compose([
    A.Resize(224, 224),
    A.RandomBrightnessContrast(
        brightness_limit=0.30, contrast_limit=0.25, p=0.8
    ),
    A.HueSaturationValue(
        hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=10, p=0.6
    ),
    _make_image_compression(p=0.4),
    A.Blur(blur_limit=(3, 5), p=0.3),
    _make_gauss_noise(),
])

# ─────────────────────────────────────────────────────────────
# STRATEGY 3 — MixUp (offline version for scarce classes)
# ─────────────────────────────────────────────────────────────

def mixup_images(img1: np.ndarray, img2: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """Blend two images at a random Beta(alpha, alpha) ratio."""
    lam = np.random.beta(alpha, alpha)
    h, w = img1.shape[:2]
    img2_resized = cv2.resize(img2, (w, h))
    blended = (lam * img1.astype(np.float32) +
               (1 - lam) * img2_resized.astype(np.float32))
    return np.clip(blended, 0, 255).astype(np.uint8)


def cutmix_images(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """Paste a random rectangular crop of img2 onto img1."""
    h, w = img1.shape[:2]
    img2_resized = cv2.resize(img2, (w, h))
    lam   = np.random.uniform(0.3, 0.7)
    cut_w = int(w * np.sqrt(1 - lam))
    cut_h = int(h * np.sqrt(1 - lam))
    cx    = np.random.randint(0, w)
    cy    = np.random.randint(0, h)
    x1    = np.clip(cx - cut_w // 2, 0, w)
    x2    = np.clip(cx + cut_w // 2, 0, w)
    y1    = np.clip(cy - cut_h // 2, 0, h)
    y2    = np.clip(cy + cut_h // 2, 0, h)
    result = img1.copy()
    result[y1:y2, x1:x2] = img2_resized[y1:y2, x1:x2]
    return result

# ─────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────

def load_image(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path))
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_image(img: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def get_image_paths(folder: Path) -> list[Path]:
    return [p for p in folder.iterdir()
            if p.suffix.lower() in IMG_EXTENSIONS]


def count_class_images(data_dir: Path, split: str = "train") -> dict[str, int]:
    counts = {}
    for cls in CLASSES:
        folder = data_dir / split / cls
        counts[cls] = len(get_image_paths(folder)) if folder.exists() else 0
    return counts

# ─────────────────────────────────────────────────────────────
# STEP A — COPY SYNTHETIC IMAGES INTO DATASET
# ─────────────────────────────────────────────────────────────

def intake_synthetic_images(synthetic_dir: Path, output_train: Path):
    """
    Copy vet-verified synthetic images from synthetic_intake/ into
    the training split. These are labeled as synthetic in filename.
    """
    if not synthetic_dir.exists():
        print(f"[SYNTHETIC] No intake folder found at '{synthetic_dir}' — skipping.")
        return

    total = 0
    for cls in CLASSES:
        src_folder = synthetic_dir / cls
        if not src_folder.exists():
            continue
        dst_folder = output_train / cls
        dst_folder.mkdir(parents=True, exist_ok=True)
        images = get_image_paths(src_folder)
        for i, img_path in enumerate(images):
            dst = dst_folder / f"synthetic_{cls}_{i:04d}{img_path.suffix}"
            img = cv2.imread(str(img_path))
            if img is not None:
                cv2.imwrite(str(dst), img)
                total += 1
        if images:
            print(f"  [SYNTHETIC] {cls}: copied {len(images)} synthetic images")
    print(f"[SYNTHETIC] Total synthetic images ingested: {total}\n")

# ─────────────────────────────────────────────────────────────
# STEP B — GEOMETRIC + PHOTOMETRIC AUGMENTATION
# ─────────────────────────────────────────────────────────────

def augment_split(
    src_dir: Path,
    dst_dir: Path,
    split: str,
    geo_n: int = GEO_MULTIPLIER,
    photo_n: int = PHOTO_MULTIPLIER,
):
    """
    For each image in src_dir/split/cls/:
      - Copy original
      - Generate geo_n geometric variants
      - Generate photo_n photometric variants
    val and test splits: copy originals only (no augmentation).
    """
    print(f"\n[AUG] Processing split: {split}")
    augment = (split == "train")

    for cls in CLASSES:
        src_cls = src_dir / split / cls
        dst_cls = dst_dir / split / cls
        dst_cls.mkdir(parents=True, exist_ok=True)

        if not src_cls.exists():
            print(f"  [{cls}] source folder missing — skipped")
            continue

        images = get_image_paths(src_cls)
        if not images:
            print(f"  [{cls}] 0 images — skipped")
            continue

        saved = 0
        for img_path in tqdm(images, desc=f"  {cls}", leave=False):
            img = load_image(img_path)
            if img is None:
                continue

            stem = img_path.stem
            ext  = img_path.suffix

            # Always copy the original
            save_image(img, dst_cls / f"{stem}_orig{ext}")
            saved += 1

            if not augment:
                continue

            # Geometric variants
            for gi in range(geo_n):
                aug = geo_transform(image=img)["image"]
                save_image(aug, dst_cls / f"{stem}_geo{gi}{ext}")
                saved += 1

            # Photometric variants
            for pi in range(photo_n):
                aug = photo_transform(image=img)["image"]
                save_image(aug, dst_cls / f"{stem}_photo{pi}{ext}")
                saved += 1

        label = "augmented" if augment else "copied"
        print(f"  [{cls}] {len(images)} source → {saved} {label}")

# ─────────────────────────────────────────────────────────────
# STEP C — MixUp / CutMix for scarce classes
# ─────────────────────────────────────────────────────────────

def apply_mixup_cutmix(
    aug_train_dir: Path,
    counts: dict[str, int],
    n_copies: int = MIXUP_MULTIPLIER,
    alpha: float  = 0.2,
):
    """
    For classes below SCARCE_THRESHOLD, generate n_copies extra images
    using MixUp and CutMix by pairing random images within the same class.
    """
    print("\n[MIXUP/CUTMIX] Processing scarce classes...")

    for cls in CLASSES:
        cls_dir = aug_train_dir / "train" / cls
        n = counts.get(cls, 0)

        if n >= SCARCE_THRESHOLD:
            print(f"  [{cls}] {n} images — above threshold, skipping")
            continue

        images = get_image_paths(cls_dir)
        if len(images) < 2:
            print(f"  [{cls}] fewer than 2 images — cannot mix, skipping")
            continue

        print(f"  [{cls}] {n} images — generating {n_copies * len(images)} mixed copies")

        for i, img_path in enumerate(tqdm(images, desc=f"  {cls} mix", leave=False)):
            img1 = load_image(img_path)
            if img1 is None:
                continue

            for j in range(n_copies):
                # Pick a random different image from the same class
                partner_path = random.choice(images)
                img2 = load_image(partner_path)
                if img2 is None:
                    continue

                # Alternate MixUp and CutMix
                if j % 2 == 0:
                    result = mixup_images(img1, img2, alpha=alpha)
                    mode   = "mixup"
                else:
                    result = cutmix_images(img1, img2)
                    mode   = "cutmix"

                out_name = f"{img_path.stem}_{mode}_{j:02d}{img_path.suffix}"
                save_image(result, cls_dir / out_name)

# ─────────────────────────────────────────────────────────────
# STEP D — WEIGHTED SAMPLER (for DataLoader)
# ─────────────────────────────────────────────────────────────

def build_weighted_sampler(
    dataset,                          # torchvision ImageFolder dataset
    minority_boost: float = 4.0,      # extra weight multiplier for scarce classes
) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler so minority classes are sampled
    at minority_boost× the rate of majority classes per epoch.

    Usage:
        dataset = ImageFolder("augmented_dataset/train", transform=...)
        sampler = build_weighted_sampler(dataset, minority_boost=4.0)
        loader  = DataLoader(dataset, batch_size=32, sampler=sampler)
    """
    class_counts = np.bincount(dataset.targets)
    class_weights = 1.0 / (class_counts + 1e-6)

    # Extra boost for very small classes
    for idx, cls_name in enumerate(dataset.classes):
        if class_counts[idx] < SCARCE_THRESHOLD:
            class_weights[idx] *= minority_boost

    sample_weights = [class_weights[t] for t in dataset.targets]
    sampler = WeightedRandomSampler(
        weights     = torch.DoubleTensor(sample_weights),
        num_samples = len(dataset),
        replacement = True,
    )
    print(f"\n[SAMPLER] Class weights (normalised):")
    for idx, cls_name in enumerate(dataset.classes):
        print(f"  {cls_name:15s} count={class_counts[idx]:6d}  weight={class_weights[idx]:.4f}")
    return sampler


# ─────────────────────────────────────────────────────────────
# TRAINING TRANSFORMS (use these in your model training script)
# ─────────────────────────────────────────────────────────────

train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.RandomBrightnessContrast(0.25, 0.20, p=0.6),
    A.HueSaturationValue(10, 25, 10, p=0.5),
    _make_image_compression(p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# ─────────────────────────────────────────────────────────────
# DATASET CLASS with online MixUp for training
# ─────────────────────────────────────────────────────────────

class CattleDiseaseDataset(Dataset):
    """
    Custom dataset with ONLINE MixUp augmentation during training.
    Pass mixup_alpha=0.0 to disable MixUp (for val/test).
    """

    def __init__(
        self,
        root: str,
        transform=None,
        mixup_alpha: float = 0.2,
        mixup_prob:  float = 0.3,     # probability of applying MixUp to a batch
    ):
        from torchvision.datasets import ImageFolder
        self.dataset     = ImageFolder(root, transform=None)
        self.transform   = transform
        self.mixup_alpha = mixup_alpha
        self.mixup_prob  = mixup_prob
        self.classes     = self.dataset.classes
        self.targets     = self.dataset.targets

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path, label = self.dataset.samples[idx]
        img = load_image(Path(img_path))
        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.uint8)

        # Online MixUp with probability mixup_prob
        if self.mixup_alpha > 0 and random.random() < self.mixup_prob:
            idx2       = random.randint(0, len(self.dataset) - 1)
            img2_path, label2 = self.dataset.samples[idx2]
            img2 = load_image(Path(img2_path))
            if img2 is not None:
                img = mixup_images(img, img2, alpha=self.mixup_alpha)

        if self.transform:
            img = self.transform(image=img)["image"]

        return img, label


# ─────────────────────────────────────────────────────────────
# FOCAL LOSS (for class imbalance)
# ─────────────────────────────────────────────────────────────

class FocalLoss(torch.nn.Module):
    """
    Focal Loss — down-weights easy examples, focuses on hard/rare ones.
    gamma=2.0, alpha=None (or pass per-class weights as alpha tensor).
    """

    def __init__(self, gamma: float = 2.0, alpha=None, reduction: str = "mean"):
        super().__init__()
        self.gamma     = gamma
        self.alpha     = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = torch.nn.functional.cross_entropy(
            inputs, targets, weight=self.alpha, reduction="none"
        )
        pt      = torch.exp(-ce_loss)
        focal   = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal


def compute_class_weights(counts: dict[str, int]) -> torch.Tensor:
    """Inverse-frequency weights for FocalLoss alpha."""
    total  = sum(counts.values())
    n_cls  = len(counts)
    weights = []
    for cls in CLASSES:
        c = counts.get(cls, 1)
        weights.append(total / (n_cls * c))
    w = torch.tensor(weights, dtype=torch.float32)
    return w / w.sum() * n_cls     # normalise so mean weight ≈ 1


# ─────────────────────────────────────────────────────────────
# SUMMARY REPORT
# ─────────────────────────────────────────────────────────────

def print_summary(src_dir: Path, dst_dir: Path):
    print("\n" + "="*60)
    print("AUGMENTATION SUMMARY")
    print("="*60)
    print(f"{'Class':<15} {'Before (train)':>16} {'After (train)':>14}")
    print("-"*48)
    for cls in CLASSES:
        before = len(get_image_paths(src_dir / "train" / cls)) \
                 if (src_dir / "train" / cls).exists() else 0
        after  = len(get_image_paths(dst_dir / "train" / cls)) \
                 if (dst_dir / "train" / cls).exists() else 0
        ratio  = f"({after // max(before,1)}×)" if before else ""
        print(f"  {cls:<13} {before:>14,}   {after:>12,}  {ratio}")
    print("="*60)

    # Write labels.txt
    labels_path = dst_dir / "labels.txt"
    with open(labels_path, "w") as f:
        for cls in CLASSES:
            f.write(cls + "\n")
    print(f"\nLabels file written: {labels_path}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Cattle disease augmentation pipeline"
    )
    p.add_argument("--data_dir",      default="final_dataset",
                   help="Root of the merged dataset (with train/val/test)")
    p.add_argument("--output_dir",    default="augmented_dataset",
                   help="Where to write augmented images")
    p.add_argument("--synthetic_dir", default=SYNTHETIC_DIR,
                   help="Folder with vet-verified synthetic images")
    p.add_argument("--geo_n",         type=int, default=GEO_MULTIPLIER)
    p.add_argument("--photo_n",       type=int, default=PHOTO_MULTIPLIER)
    p.add_argument("--mixup_n",       type=int, default=MIXUP_MULTIPLIER)
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--skip_val_test", action="store_true",
                   help="Skip copying val and test splits (faster for re-runs)")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    syn_dir    = Path(args.synthetic_dir)

    print("="*60)
    print("CATTLE DISEASE — AUGMENTATION PIPELINE")
    print("="*60)
    print(f"  Source      : {data_dir.resolve()}")
    print(f"  Output      : {output_dir.resolve()}")
    print(f"  Synthetic   : {syn_dir.resolve()}")
    print(f"  Geo ×{args.geo_n}  Photo ×{args.photo_n}  MixUp ×{args.mixup_n}")
    print()

    # ── Pre-flight check ───────────────────────────────────────
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir not found: {data_dir}")

    before_counts = count_class_images(data_dir, "train")
    print("Pre-augmentation train counts:")
    for cls, n in before_counts.items():
        flag = " ← SCARCE" if n < SCARCE_THRESHOLD else ""
        print(f"  {cls:<15} {n:>6,}{flag}")

    # ── Step A: Synthetic intake ───────────────────────────────
    print("\n" + "─"*40)
    print("STEP A — Synthetic image intake")
    print("─"*40)
    intake_synthetic_images(syn_dir, data_dir / "train")

    # Refresh counts after synthetic intake
    before_counts = count_class_images(data_dir, "train")

    # ── Step B: Geometric + Photometric augmentation ───────────
    print("\n" + "─"*40)
    print("STEP B — Geometric + Photometric augmentation")
    print("─"*40)

    splits_to_process = ["train"]
    if not args.skip_val_test:
        splits_to_process += ["val", "test"]

    for split in splits_to_process:
        augment_split(
            src_dir  = data_dir,
            dst_dir  = output_dir,
            split    = split,
            geo_n    = args.geo_n,
            photo_n  = args.photo_n,
        )

    # ── Step C: MixUp / CutMix for scarce classes ──────────────
    print("\n" + "─"*40)
    print("STEP C — MixUp / CutMix (scarce classes)")
    print("─"*40)
    apply_mixup_cutmix(
        aug_train_dir = output_dir,
        counts        = before_counts,
        n_copies      = args.mixup_n,
    )

    # ── Summary ────────────────────────────────────────────────
    print_summary(data_dir, output_dir)

    print("""
STEP D — Weighted sampler (use in your training script):
─────────────────────────────────────────────────────────
from torchvision.datasets import ImageFolder
from augmentation_pipeline import (
    build_weighted_sampler, CattleDiseaseDataset,
    FocalLoss, compute_class_weights,
    train_transform, val_transform
)

# Training loader with weighted sampler
train_ds  = CattleDiseaseDataset(
    "augmented_dataset/train",
    transform=train_transform,
    mixup_alpha=0.2, mixup_prob=0.3
)
sampler   = build_weighted_sampler(train_ds, minority_boost=4.0)
train_loader = DataLoader(train_ds, batch_size=32,
                          sampler=sampler, num_workers=4)

# Validation loader (no augmentation, no sampler)
val_ds    = CattleDiseaseDataset(
    "augmented_dataset/val",
    transform=val_transform, mixup_alpha=0.0
)
val_loader = DataLoader(val_ds, batch_size=32,
                        shuffle=False, num_workers=4)

# Focal loss with inverse-frequency class weights
class_counts = {cls: n for cls, n in zip(CLASSES, [12,1114,8550,6742,4358,6])}
alpha        = compute_class_weights(class_counts)
criterion    = FocalLoss(gamma=2.0, alpha=alpha)
─────────────────────────────────────────────────────────
""")


if __name__ == "__main__":
    main()