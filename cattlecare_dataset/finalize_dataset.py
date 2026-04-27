"""
=============================================================
 finalize_dataset.py
=============================================================
 PURPOSE:
   augmented_dataset/train has all the images, but:
     - BRD, Mastitis, Orf have 0 val and 0 test images
     - The albumentations v2 API warnings need fixing
     - We need ONE clean, validated dataset ready for training

 WHAT THIS SCRIPT DOES:
   1. Reads augmented_dataset/train for every class
   2. Carves out val (15%) and test (10%) splits STRATIFIED
      (so every class gets proper val/test, even scarce ones)
   3. Copies existing val/test from augmented_dataset where
      they already exist (FMD, Healthy_Cow, LSD)
   4. Writes everything to: ready_dataset/train|val|test/cls/
   5. Prints a full validation report

 SPLIT STRATEGY for scarce classes (BRD=180, Orf=90):
   - BRD  180 total -> train=144  val=18  test=18
   - Orf   90 total -> train=72   val=9   test=9
   (These are augmented images - enough for validation now)

 Usage:
   python finalize_dataset.py
   python finalize_dataset.py --src augmented_dataset --dst ready_dataset
   python finalize_dataset.py --val_pct 0.15 --test_pct 0.10

 Requirements: pip install tqdm
=============================================================
"""

import os
import shutil
import random
import argparse
from pathlib import Path
from collections import defaultdict
import hashlib
import time
import stat
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

CLASSES        = ["BRD", "FMD", "Healthy_Cow", "LSD", "Mastitis", "Orf"]
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Classes that already have val/test in augmented_dataset
# -> use those directly instead of carving from train
PREBUILT_VAL_TEST = {"FMD", "Healthy_Cow", "LSD"}

# Minimum images to keep in val and test per class
MIN_VAL  = 8
MIN_TEST = 8


# ─────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────

def get_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir()
                   if p.suffix.lower() in IMG_EXTENSIONS])


def copy_images(src_list: list[Path], dst_folder: Path, desc: str = ""):
    dst_folder.mkdir(parents=True, exist_ok=True)
    for src in tqdm(src_list, desc=f"  {desc}", leave=False):
        shutil.copy2(src, dst_folder / src.name)

def _rmtree_onerror(func, path, exc_info):
    # Windows can fail deletes due to readonly attributes
    try:
        os.chmod(path, stat.S_IWRITE)
    except Exception:
        pass
    try:
        func(path)
    except Exception:
        pass


def safe_rmtree(path: Path, retries: int = 5, delay_s: float = 0.5):
    """
    Robust directory removal for Windows.
    Retries because antivirus / indexing / just-killed processes can keep handles briefly.
    """
    if not path.exists():
        return
    last_err: Exception | None = None
    for _ in range(retries):
        try:
            shutil.rmtree(path, onerror=_rmtree_onerror)
            return
        except Exception as e:
            last_err = e
            time.sleep(delay_s)
    if last_err:
        raise last_err

def file_signature(path: Path) -> str:
    """
    Fast content signature for dedupe across splits.
    We avoid full-file hashing because datasets can be huge.

    Signature uses:
      - file size
      - md5(first 8KB + last 8KB)
    This is extremely unlikely to collide for real image files and is
    plenty to prevent cross-split leakage.
    """
    st = path.stat()
    size = st.st_size
    h = hashlib.md5()
    with open(path, "rb") as f:
        head = f.read(8192)
        if size > 8192:
            try:
                f.seek(max(0, size - 8192))
            except Exception:
                pass
            tail = f.read(8192)
        else:
            tail = b""
    h.update(head)
    h.update(tail)
    return f"{size}:{h.hexdigest()}"


def copy_images_deduped(
    src_list: list[Path],
    dst_folder: Path,
    seen_hashes: set[str],
    desc: str = "",
) -> tuple[int, int]:
    """
    Copy images while preventing duplicates by file content hash.
    Returns: (copied_count, skipped_duplicates_count)
    """
    dst_folder.mkdir(parents=True, exist_ok=True)
    copied = 0
    skipped = 0
    for src in tqdm(src_list, desc=f"  {desc}", leave=False):
        try:
            h = file_signature(src)
        except Exception:
            # If we can't hash it, skip; validate_final will catch corrupt files.
            skipped += 1
            continue
        if h in seen_hashes:
            skipped += 1
            continue
        seen_hashes.add(h)
        shutil.copy2(src, dst_folder / src.name)
        copied += 1
    return copied, skipped


def compute_split_sizes(
    n_total: int,
    val_pct: float,
    test_pct: float,
) -> tuple[int, int, int]:
    """Return (n_train, n_val, n_test) given total count and percentages."""
    n_val  = max(MIN_VAL,  int(n_total * val_pct))
    n_test = max(MIN_TEST, int(n_total * test_pct))
    # Don't let val+test consume more than 40% of a scarce class
    if n_val + n_test >= n_total:
        n_val  = max(1, n_total // 5)
        n_test = max(1, n_total // 10)
    n_train = n_total - n_val - n_test
    return n_train, n_val, n_test


# ─────────────────────────────────────────────────────────────
# MAIN LOGIC
# ─────────────────────────────────────────────────────────────

def finalize(
    src: Path,
    dst: Path,
    val_pct: float,
    test_pct: float,
    seed: int,
):
    random.seed(seed)

    print("=" * 60)
    print("FINALIZING DATASET")
    print("=" * 60)
    print(f"  Source : {src.resolve()}")
    print(f"  Output : {dst.resolve()}")
    print(f"  Val    : {val_pct*100:.0f}%   Test : {test_pct*100:.0f}%")
    print()

    # Always ensure output root exists (even if upstream is empty)
    dst.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        print(f"[!] Output dir '{dst}' already exists - clearing it first.")
        safe_rmtree(dst)
        dst.mkdir(parents=True, exist_ok=True)

    results = {}   # cls -> {train, val, test} counts

    for cls in CLASSES:
        # ASCII-only separators (Windows consoles often choke on box-drawing chars)
        print(f"-- {cls} " + "-" * (40 - len(cls)))

        train_src = src / "train" / cls
        val_src   = src / "val"   / cls
        test_src  = src / "test"  / cls

        train_imgs = get_images(train_src)

        if not train_imgs:
            print(f"  [SKIP] No training images found in {train_src}")
            results[cls] = {"train": 0, "val": 0, "test": 0}
            continue

        dst_train = dst / "train" / cls
        dst_val   = dst / "val"   / cls
        dst_test  = dst / "test"  / cls

        # Track hashes across splits to avoid leakage in the output.
        # Priority: keep val/test clean; train drops anything duplicated.
        seen_hashes: set[str] = set()

        # ── Classes with pre-built val/test (FMD, Healthy_Cow, LSD) ────
        if cls in PREBUILT_VAL_TEST:
            val_imgs  = get_images(val_src)
            test_imgs = get_images(test_src)

            if val_imgs and test_imgs:
                print(f"  Using pre-built val ({len(val_imgs)}) "
                      f"and test ({len(test_imgs)}) splits")
                # Copy val/test first, then train with dedupe against them.
                val_copied, val_skipped = copy_images_deduped(
                    val_imgs, dst_val, seen_hashes, f"{cls}/val (dedupe)"
                )
                test_copied, test_skipped = copy_images_deduped(
                    test_imgs, dst_test, seen_hashes, f"{cls}/test (dedupe)"
                )
                train_copied, train_skipped = copy_images_deduped(
                    train_imgs, dst_train, seen_hashes, f"{cls}/train (dedupe)"
                )
                if val_skipped or test_skipped or train_skipped:
                    print(
                        f"  [DEDUPED] skipped duplicates - "
                        f"train={train_skipped}, val={val_skipped}, test={test_skipped}"
                    )
                results[cls] = {
                    "train": train_copied,
                    "val":   val_copied,
                    "test":  test_copied,
                }
                continue

        # Classes without val/test - carve from train
        n_total = len(train_imgs)
        n_train, n_val, n_test = compute_split_sizes(n_total, val_pct, test_pct)

        print(f"  {n_total} train images -> "
              f"carving val={n_val}  test={n_test}  "
              f"keeping train={n_train}")

        # Shuffle with fixed seed for reproducibility
        shuffled = train_imgs.copy()
        random.shuffle(shuffled)

        test_set  = shuffled[:n_test]
        val_set   = shuffled[n_test: n_test + n_val]
        train_set = shuffled[n_test + n_val:]

        # Copy val/test first, then train, all deduped by content hash.
        val_copied, val_skipped = copy_images_deduped(
            val_set, dst_val, seen_hashes, f"{cls}/val (dedupe)"
        )
        test_copied, test_skipped = copy_images_deduped(
            test_set, dst_test, seen_hashes, f"{cls}/test (dedupe)"
        )
        train_copied, train_skipped = copy_images_deduped(
            train_set, dst_train, seen_hashes, f"{cls}/train (dedupe)"
        )
        if val_skipped or test_skipped or train_skipped:
            print(
                f"  [DEDUPED] skipped duplicates - "
                f"train={train_skipped}, val={val_skipped}, test={test_skipped}"
            )

        results[cls] = {
            "train": train_copied,
            "val":   val_copied,
            "test":  test_copied,
        }

    # ── labels.txt ───────────────────────────────────────────────────────
    labels_path = dst / "labels.txt"
    with open(labels_path, "w") as f:
        for cls in CLASSES:
            f.write(cls + "\n")

    # ── Validation report ────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("FINAL DATASET VALIDATION REPORT")
    print("=" * 65)
    print(f"{'Class':<15} {'Train':>10} {'Val':>8} {'Test':>8} {'Total':>8}")
    print("-" * 55)

    grand = defaultdict(int)
    for cls in CLASSES:
        r = results.get(cls, {"train": 0, "val": 0, "test": 0})
        total = r["train"] + r["val"] + r["test"]
        flags = ""
        if r["val"] == 0:
            flags += "  ← NO VAL"
        if r["test"] == 0:
            flags += "  ← NO TEST"
        print(f"  {cls:<13} {r['train']:>10,} {r['val']:>8,} "
              f"{r['test']:>8,} {total:>8,}{flags}")
        for k in ["train", "val", "test"]:
            grand[k] += r[k]

    total_all = grand["train"] + grand["val"] + grand["test"]
    print("-" * 55)
    print(f"  {'TOTAL':<13} {grand['train']:>10,} {grand['val']:>8,} "
          f"{grand['test']:>8,} {total_all:>8,}")
    print("=" * 65)
    print(f"\nLabels file : {labels_path}")
    print(f"Ready for   : torchvision.datasets.ImageFolder('{dst}/train')")

    # ── Imbalance warning ─────────────────────────────────────────────────
    train_counts = {cls: results[cls]["train"] for cls in CLASSES}
    max_cls = max(train_counts, key=train_counts.get)
    min_cls = min(train_counts, key=lambda c: train_counts[c] or 999999)
    ratio = train_counts[max_cls] / max(train_counts[min_cls], 1)

    print()
    if ratio > 50:
        print(f"[!] IMBALANCE RATIO = {ratio:.0f}x "
              f"({max_cls} vs {min_cls})")
        print("    -> Use WeightedRandomSampler + FocalLoss in training.")
        print("    -> See augmentation_pipeline.py for build_weighted_sampler()")
    else:
        print(f"[OK] Imbalance ratio = {ratio:.1f}x - acceptable for training.")

    print()
    print("Next step:")
    print("  python train.py --data_dir ready_dataset")
    print()


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Finalize augmented dataset with proper val/test splits"
    )
    p.add_argument("--src",      default="augmented_dataset",
                   help="Source: output of augmentation_pipeline.py")
    p.add_argument("--dst",      default="ready_dataset",
                   help="Output: clean dataset ready for training")
    p.add_argument("--val_pct",  type=float, default=0.15,
                   help="Fraction of train to carve into val (default 0.15)")
    p.add_argument("--test_pct", type=float, default=0.10,
                   help="Fraction of train to carve into test (default 0.10)")
    p.add_argument("--seed",     type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    finalize(
        src      = Path(args.src),
        dst      = Path(args.dst),
        val_pct  = args.val_pct,
        test_pct = args.test_pct,
        seed     = args.seed,
    )