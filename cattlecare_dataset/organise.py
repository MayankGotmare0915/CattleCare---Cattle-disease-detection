"""
organise_fixed.py  (v3 — fixed)
================================
Two bugs fixed vs v2:
  1. Summary now reads actual file counts from disk (not just copy-op counters),
     so images already in bucket_A_raw are correctly reported.
  2. Removed self-referencing mappings (bucket_A_raw → bucket_A_raw).
     Images placed there by classify_mdzillur.py / scrape_diseases.py
     are counted directly from disk in the summary.
  3. mdzillur flat handler now COPIES directly into bucket_A_raw/<class>
     using filename-pattern classification inline, so you don't have to
     run classify_mdzillur.py separately just to unblock organise.
"""

import shutil
from pathlib import Path

# ── YOUR BASE DIRECTORY ──────────────────────────────────────────
BASE = Path("C:/Projects/CattleCare/cattlecare_dataset")

IMG_EXT = {"*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"}

# ── EXACT PATH MAPPINGS ──────────────────────────────────────────
# Only EXTERNAL sources here.
# Images already in bucket_A_raw (from scrape_diseases / classify_mdzillur)
# are counted from disk in the summary — NOT listed as mappings.

MAPPINGS = [

    # ════════════════════════════════════════════════════════════
    # MASTITIS
    # ════════════════════════════════════════════════════════════
    (BASE/"roboflow_downloads/kirubel_mastitis/train",
     "Mastitis", "roboflow"),

    # ════════════════════════════════════════════════════════════
    # FMD
    # ════════════════════════════════════════════════════════════
    (BASE/"roboflow_downloads/moaz_fmd/train",          "FMD", "roboflow"),
    (BASE/"roboflow_downloads/moaz_fmd/test",           "FMD", "roboflow"),
    (BASE/"kaggle_downloads/cattle-diseases-datasets/Cows datasets/foot-and-mouth",
     "FMD", "raw"),

    # ════════════════════════════════════════════════════════════
    # LSD
    # ════════════════════════════════════════════════════════════
    (BASE/"roboflow_downloads/qq_lsd/train/Lumpy Skin",        "LSD", "roboflow"),
    (BASE/"roboflow_downloads/sliit_cattle/train/lumpy skin",  "LSD", "roboflow"),
    (BASE/"kaggle_downloads/cattle-diseases-datasets/Cows datasets/lumpy",
     "LSD", "raw"),
    (BASE/"kaggle_downloads/cow-lumpy-disease-dataset/lumpycows",
     "LSD", "raw"),

    # ════════════════════════════════════════════════════════════
    # HEALTHY COW
    # ════════════════════════════════════════════════════════════
    (BASE/"roboflow_downloads/qq_lsd/train/Normal Skin",       "Healthy_Cow", "roboflow"),
    (BASE/"roboflow_downloads/sliit_cattle/train/healthy",     "Healthy_Cow", "roboflow"),
    (BASE/"kaggle_downloads/cattle-diseases-datasets/Cows datasets/healthy",
     "Healthy_Cow", "raw"),
    (BASE/"kaggle_downloads/cow-lumpy-disease-dataset/healthycows",
     "Healthy_Cow", "raw"),

    # ════════════════════════════════════════════════════════════
    # BRD
    # ════════════════════════════════════════════════════════════
    (BASE/"roboflow_downloads/sliit_cattle/valid/(BRD) Bovine Disease Respiratory",
     "BRD", "roboflow"),
    (BASE/"roboflow_downloads/sliit_cattle/valid/(BRD) Bovine Dermatitis Disease healthy lumpy",
     "BRD", "roboflow"),

    # ════════════════════════════════════════════════════════════
    # ORF
    # ════════════════════════════════════════════════════════════
    (BASE/"roboflow_downloads/sliit_cattle/valid/Contagious Ecthym",
     "Orf", "roboflow"),

]

# ── FILENAME PATTERNS for inline mdzillur classification ─────────
MDZILLUR_PATTERNS = [
    ("LSD",         ["lumpy", "lsd", "lumpskin", "lump_skin", "nodule"]),
    ("Mastitis",    ["mastit", "udder", "teat", "mammary"]),
    ("FMD",         ["fmd", "foot_mouth", "footmouth", "foot-mouth", "hoof_sore"]),
    ("Healthy_Cow", ["healthy", "normal", "normalskin", "normal_skin", "healthycow"]),
    ("BRD",         ["brd", "respiratory", "pneumonia", "shipping_fever", "nasal"]),
    ("Mange",       ["mange", "scabies", "mite", "sarcoptic", "chorioptic"]),
    ("Orf",         ["orf", "ecthyma", "ecthym"]),
    ("Ringworm",    ["ringworm", "dermatophyte", "ring_worm"]),
]

def classify_filename(filename: str) -> str:
    name = filename.lower()
    original = name.split(".rf.")[0] if ".rf." in name else name
    for cls, patterns in MDZILLUR_PATTERNS:
        for p in patterns:
            if p in original:
                return cls
    return "_unclassified"


# ── COPY FUNCTION ────────────────────────────────────────────────
def count_images_in(folder: Path) -> list:
    imgs = []
    for ext in IMG_EXT:
        imgs.extend(folder.glob(ext))
    return imgs

def copy_images(src_path, dest_class, bucket):
    src = Path(src_path)
    if not src.exists():
        return 0, "not_found"

    dest_dir = BASE / ("bucket_A_raw" if bucket == "raw" else "bucket_B_roboflow") / dest_class
    dest_dir.mkdir(parents=True, exist_ok=True)

    images = count_images_in(src)
    if not images:
        return 0, "empty"

    source_tag = src.parent.parent.name
    copied = 0
    for img_path in images:
        dest_file = dest_dir / f"{source_tag}__{img_path.name}"
        if not dest_file.exists():
            shutil.copy2(img_path, dest_file)
            copied += 1
    return copied, "ok"


# ── COUNT WHAT'S ACTUALLY ON DISK ────────────────────────────────
def count_bucket(bucket_dir: Path) -> dict:
    """Return {class_name: image_count} by reading disk directly."""
    counts = {}
    if not bucket_dir.exists():
        return counts
    for cls_dir in bucket_dir.iterdir():
        if not cls_dir.is_dir() or cls_dir.name.startswith("_"):
            continue
        n = len(count_images_in(cls_dir))
        if n > 0:
            counts[cls_dir.name] = n
    return counts


# ── MDZILLUR FLAT HANDLER ────────────────────────────────────────
def handle_mdzillur(copy_counters: dict):
    """
    If mdzillur/train is still flat, classify images by filename pattern
    and copy directly into bucket_A_raw/<class>/.
    Counts are added to copy_counters["raw"][class].
    """
    print("\n" + "-"*60)
    mdzillur_train = BASE / "roboflow_downloads/mdzillur_cattle/train"

    if not mdzillur_train.exists():
        print("  mdzillur_cattle/train — not found, skipping")
        return

    subfolders = [d for d in mdzillur_train.iterdir() if d.is_dir()]

    if subfolders:
        # classify_mdzillur.py already made subfolders — use them
        print(f"  mdzillur has {len(subfolders)} subfolders — copying by subfolder name")
        SUBFOLDER_MAP = {
            "lumpy": "LSD", "lsd": "LSD",
            "mastitis": "Mastitis",
            "fmd": "FMD",
            "healthy": "Healthy_Cow", "normal": "Healthy_Cow",
            "brd": "BRD",
            "mange": "Mange",
            "ringworm": "Ringworm",
            "orf": "Orf",
        }
        for sub in sorted(subfolders):
            key    = sub.name.lower()
            mapped = next((v for k, v in SUBFOLDER_MAP.items() if k in key), None)
            if not mapped:
                print(f"    UNMAPPED subfolder: {sub.name}")
                continue
            n, _ = copy_images(sub, mapped, "roboflow")
            if n:
                print(f"    ✓ {n:>5}  {sub.name} → {mapped}")
                copy_counters["roboflow"][mapped] = copy_counters["roboflow"].get(mapped, 0) + n
        return

    # Flat — classify by filename pattern inline
    images = count_images_in(mdzillur_train)
    print(f"  mdzillur flat ({len(images)} images) — classifying by filename pattern...")

    class_counts = {}
    for img_path in images:
        cls      = classify_filename(img_path.name)
        dest_dir = RAW_BUCKET / cls
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / f"mdzillur__{img_path.name}"
        if not dest_file.exists():
            shutil.copy2(img_path, dest_file)
        class_counts[cls] = class_counts.get(cls, 0) + 1

    for cls, n in sorted(class_counts.items()):
        mark = "⚠" if cls == "_unclassified" else "✓"
        print(f"    {mark} {n:>5}  → {cls}")
        copy_counters["raw"][cls] = copy_counters["raw"].get(cls, 0) + n

    unclassified = class_counts.get("_unclassified", 0)
    if unclassified:
        print(f"\n  {unclassified} images could not be classified by filename.")
        print(f"  Review them in: {RAW_BUCKET / '_unclassified'}")
        print(f"  Then run:  python classify_mdzillur.py --review")


# ── MAIN ─────────────────────────────────────────────────────────
RAW_BUCKET      = BASE / "bucket_A_raw"
ROBOFLOW_BUCKET = BASE / "bucket_B_roboflow"

print("="*60)
print("ORGANISING IMAGES INTO BUCKETS (v3)")
print("="*60)

# copy_counters tracks only NEW copies made this run
copy_counters = {"raw": {}, "roboflow": {}}

for src_path, dest_class, bucket in MAPPINGS:
    n, status = copy_images(src_path, dest_class, bucket)
    if status == "not_found":
        pass   # silent — most kaggle/roboflow paths are optional
    elif n > 0:
        label = "raw" if bucket == "raw" else "roboflow"
        print(f"  ✓ {n:>5}  →  {dest_class:<20} [{label}]")
        copy_counters[bucket][dest_class] = copy_counters[bucket].get(dest_class, 0) + n

# Handle mdzillur (flat or subfoldered)
handle_mdzillur(copy_counters)

# Preserve existing raw_images/ if present
raw_images_dir = BASE / "raw_images"
if raw_images_dir.exists():
    print("\n" + "-"*60)
    print("Preserving raw_images/...")
    for cls_dir in raw_images_dir.iterdir():
        if not cls_dir.is_dir():
            continue
        imgs = count_images_in(cls_dir)
        dest = RAW_BUCKET / cls_dir.name
        dest.mkdir(parents=True, exist_ok=True)
        cp = 0
        for img in imgs:
            d = dest / img.name
            if not d.exists():
                shutil.copy2(img, d)
                cp += 1
        if cp:
            print(f"  ✓ {cp:>5}  →  {cls_dir.name:<20} [raw]")

# ── FINAL SUMMARY: read from disk, not copy counters ─────────────
print("\n" + "="*60)
print("FINAL SUMMARY  (actual disk counts)")
print("="*60)

raw_counts      = count_bucket(RAW_BUCKET)
roboflow_counts = count_bucket(ROBOFLOW_BUCKET)
all_classes     = sorted(set(list(raw_counts) + list(roboflow_counts)))

# Also include classes that are in MAPPINGS but happen to be empty
for _, dest_class, _ in MAPPINGS:
    if dest_class not in all_classes:
        all_classes.append(dest_class)
all_classes = sorted(set(all_classes))

print(f"\n  {'Class':<18}  {'bucket_A_raw':>12}  {'bucket_B_robof':>14}  {'Total':>7}  Status")
print("  " + "-"*70)

grand_raw = grand_rob = 0
for cls in all_classes:
    r = raw_counts.get(cls, 0)
    b = roboflow_counts.get(cls, 0)
    t = r + b
    grand_raw += r
    grand_rob += b
    if t >= 2000:   status = "OK"
    elif t >= 500:  status = "LOW — needs augmentation"
    elif t > 0:     status = "VERY LOW"
    else:           status = "NO DATA ✗"
    bar = "█" * min(t // 100, 25)
    print(f"  {cls:<18}  {r:>12}  {b:>14}  {t:>7}  {status}  {bar}")

print("  " + "-"*70)
print(f"  {'TOTAL':<18}  {grand_raw:>12}  {grand_rob:>14}  {grand_raw+grand_rob:>7}")

# Advice
no_data = [c for c in all_classes if raw_counts.get(c,0)+roboflow_counts.get(c,0) == 0]
low     = [c for c in all_classes if 0 < raw_counts.get(c,0)+roboflow_counts.get(c,0) < 500]

if no_data:
    print(f"\n  Classes with NO DATA: {', '.join(no_data)}")
    print("    → python scrape_diseases.py")
if low:
    print(f"\n  Classes with < 500 images: {', '.join(low)}")
    print("    → python scrape_diseases.py --class <Name> --count 400")

print("\nNext steps:")
print("  python count_and_plan.py --target 2000 --generate-aug")
print("  python diagnose.py")