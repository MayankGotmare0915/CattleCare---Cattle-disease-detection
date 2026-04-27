"""
classify_mdzillur.py  (NO-API version)
=======================================
Classifies the 3,294 flat mdzillur_cattle images WITHOUT any API calls.

STRATEGY (in order of reliability):
  1. Filename pattern matching  — Roboflow filenames often encode the class
     in the original image ID (e.g. "lumpy_cow_0012.jpg")
  2. CLIP zero-shot (optional)  — if torch + open_clip installed, runs locally
     on your GPU/CPU, completely free, no internet after install
  3. Manual review helper       — opens a Tkinter viewer for anything
     still unclassified

OUTPUT:
  bucket_A_raw/
    Mastitis/      ← classified mdzillur images
    LSD/
    FMD/
    Healthy_Cow/
    BRD/
    Mange/
    Orf/
    Ringworm/
    _unclassified/ ← review these manually

  classify_mdzillur_log.csv   ← full audit trail

INSTALL (all free, no API):
  # For CLIP (strongly recommended — much better accuracy):
  pip install torch torchvision open_clip_torch Pillow tqdm

  # Minimum (filename matching only):
  pip install Pillow tqdm

USAGE:
  python classify_mdzillur.py                  # filename match + CLIP if available
  python classify_mdzillur.py --no-clip        # filename match only (fastest)
  python classify_mdzillur.py --review         # open Tkinter viewer for _unclassified
  python classify_mdzillur.py --sample 50      # test on first 50 images
  python classify_mdzillur.py --resume         # skip already-logged images
  python classify_mdzillur.py --dry-run        # count images, no file ops
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE         = Path("C:/Projects/CattleCare/cattlecare_dataset")
MDZILLUR_DIR = BASE / "roboflow_downloads/mdzillur_cattle/train"
RAW_BUCKET   = BASE / "bucket_A_raw"
LOG_FILE     = BASE / "classify_mdzillur_log.csv"

VALID_CLASSES = [
    "Mastitis",
    "LSD",
    "FMD",
    "Healthy_Cow",
    "BRD",
    "Mange",
    "Orf",
    "Ringworm",
    "_unclassified",
]

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG", ".PNG"}

# ── STEP 1: FILENAME PATTERN MATCHING ────────────────────────────────────────
# Roboflow filenames encode the original dataset image name before ".rf."
# Example: lumpy_cow_field_023_jpg.rf.jLhmGr8O2GPS3YFgxTxL.jpg
#          → original part "lumpy_cow_field_023" contains "lumpy" → LSD
#
# These patterns are tried against the part BEFORE the ".rf." separator,
# then against the full filename if no separator found.

FILENAME_PATTERNS = {
    "LSD": [
        "lumpy", "lsd", "lumpskin", "lump_skin", "lump-skin",
        "nodule", "lumpycow",
    ],
    "Mastitis": [
        "mastit", "udder", "teat", "mammary", "mastitis",
    ],
    "FMD": [
        "fmd", "foot_mouth", "footmouth", "foot-mouth",
        "aphthous", "blister", "hoof_sore",
    ],
    "Healthy_Cow": [
        "healthy", "normal", "clean_cow", "no_disease",
        "normalskin", "normal_skin", "healthycow",
    ],
    "BRD": [
        "brd", "respiratory", "pneumonia", "shipping_fever",
        "nasal", "bovine_resp",
    ],
    "Mange": [
        "mange", "scabies", "mite", "sarcoptic", "chorioptic",
        "psoroptic", "crusty_skin",
    ],
    "Orf": [
        "orf", "ecthyma", "contagious_ecthym", "ecthym",
    ],
    "Ringworm": [
        "ringworm", "dermatophyte", "fungal_skin", "ring_worm",
    ],
}


def classify_by_filename(filename: str) -> tuple:
    """
    Match filename against known patterns.
    Returns (class_name, method_string).
    """
    name_lower = filename.lower()

    # Extract the original-name part before Roboflow's .rf. hash
    if ".rf." in name_lower:
        original_part = name_lower.split(".rf.")[0]
    else:
        original_part = name_lower

    for cls, patterns in FILENAME_PATTERNS.items():
        for pattern in patterns:
            if pattern in original_part:
                return cls, f"filename:{pattern}"

    return "_unclassified", "no_filename_match"


# ── STEP 2: CLIP ZERO-SHOT (LOCAL, FREE) ─────────────────────────────────────
# CLIP matches images to text prompts using a pre-trained neural network.
# Runs entirely on your machine. No API. No billing.
# Install: pip install torch open_clip_torch

CLIP_PROMPTS = {
    "Mastitis": [
        "a cow udder with mastitis inflammation",
        "swollen infected dairy cow teat with mastitis",
        "bovine mastitis milk disease",
    ],
    "LSD": [
        "a cow with lumpy skin disease raised nodules",
        "cattle with multiple bumps lumps on skin",
        "bovine lumpy skin disease lesions all over body",
    ],
    "FMD": [
        "foot and mouth disease blisters on cow hoof",
        "cattle with mouth sores foot and mouth disease",
        "bovine FMD vesicles blister",
    ],
    "Healthy_Cow": [
        "a healthy cow with normal clean skin",
        "healthy dairy cattle no visible disease",
        "normal bovine skin no lesions",
    ],
    "BRD": [
        "sick cow with nasal discharge bovine respiratory disease",
        "cattle pneumonia sick calf laboured breathing",
        "bovine respiratory disease runny nose",
    ],
    "Mange": [
        "cow with mange crusty scabby skin patches hair loss",
        "cattle skin mite infestation scabs",
        "bovine sarcoptic mange dermatitis",
    ],
    "Orf": [
        "cattle with orf contagious ecthyma lesions on muzzle",
        "cow lip wart-like orf disease",
    ],
    "Ringworm": [
        "cow with circular crusty ringworm skin patches",
        "bovine ringworm fungal skin lesion circle",
    ],
}

_clip_model   = None
_clip_preproc = None
_clip_device  = None
_clip_feats   = None


def load_clip() -> bool:
    """Load CLIP model once. Returns True if successful."""
    global _clip_model, _clip_preproc, _clip_device, _clip_feats

    if _clip_model is not None:
        return True

    try:
        import torch
        import open_clip
        import torch.nn.functional as F

        _clip_device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Loading CLIP (ViT-B-32) on {_clip_device}...")

        _clip_model, _, _clip_preproc = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        _clip_model.eval()
        _clip_model = _clip_model.to(_clip_device)

        tokenizer = open_clip.get_tokenizer("ViT-B-32")

        # Pre-compute text features for all prompts
        all_texts  = []
        text_class = []
        for cls, prompts in CLIP_PROMPTS.items():
            for p in prompts:
                all_texts.append(p)
                text_class.append(cls)

        with torch.no_grad():
            tokens = tokenizer(all_texts).to(_clip_device)
            feats  = _clip_model.encode_text(tokens)
            feats  = F.normalize(feats, dim=-1)

        _clip_feats = (feats, text_class)
        print("  CLIP ready.\n")
        return True

    except ImportError:
        return False
    except Exception as e:
        print(f"  CLIP load failed: {e}")
        return False


def classify_by_clip(img_path: Path) -> tuple:
    """
    Returns (class_name, confidence_float, method_string).
    """
    import torch
    import torch.nn.functional as F
    from PIL import Image

    feats, text_class = _clip_feats

    with torch.no_grad():
        img        = Image.open(img_path).convert("RGB")
        img_tensor = _clip_preproc(img).unsqueeze(0).to(_clip_device)
        img_feat   = _clip_model.encode_image(img_tensor)
        img_feat   = F.normalize(img_feat, dim=-1)
        sims       = (img_feat @ feats.T).squeeze(0).cpu().tolist()

    # Aggregate: best prompt score per class
    class_scores = {}
    for score, cls in zip(sims, text_class):
        if cls not in class_scores or score > class_scores[cls]:
            class_scores[cls] = score

    best_cls   = max(class_scores, key=class_scores.get)
    best_score = class_scores[best_cls]
    sorted_scores = sorted(class_scores.values(), reverse=True)
    gap = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 1.0

    # Mark as unclassified if confidence too low
    if best_score < 0.22 or gap < 0.02:
        return "_unclassified", best_score, f"clip_lowconf:{best_score:.3f}"

    return best_cls, best_score, f"clip:{best_score:.3f}_gap:{gap:.3f}"


# ── STEP 3: MANUAL TKINTER REVIEW ────────────────────────────────────────────
def run_manual_review():
    """
    Opens a Tkinter viewer for images in _unclassified/.
    Number keys assign a class; S skips; Q quits.
    """
    unclassified_dir = RAW_BUCKET / "_unclassified"
    images = sorted([
        p for p in unclassified_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]) if unclassified_dir.exists() else []

    if not images:
        print("No images in _unclassified/ — nothing to review.")
        return

    try:
        import tkinter as tk
        from PIL import Image, ImageTk
    except ImportError:
        print("Pillow or Tkinter not available.")
        print(f"Open this folder manually: {unclassified_dir}")
        return

    REVIEW_CLASSES = [c for c in VALID_CLASSES if c != "_unclassified"]
    shortcuts = {str(i + 1): cls for i, cls in enumerate(REVIEW_CLASSES)}

    root = tk.Tk()
    root.title("Manual Review — mdzillur unclassified")
    root.geometry("920x720")

    idx   = [0]
    moved = [0]

    info_lbl = tk.Label(root, text="", font=("Arial", 11), wraplength=880)
    info_lbl.pack(pady=4)

    canvas = tk.Label(root)
    canvas.pack(expand=True)

    key_text = (
        "\n".join(f"  {k} → {v}" for k, v in shortcuts.items())
        + "\n  S → Skip   Q → Quit"
    )
    tk.Label(root, text=key_text, font=("Courier", 10), justify="left",
             bg="#f0f0f0", relief="groove", padx=8, pady=6).pack(pady=4)

    status_lbl = tk.Label(root, text="", font=("Arial", 10))
    status_lbl.pack()

    def show():
        img_path = images[idx[0]]
        im = Image.open(img_path).convert("RGB")
        im.thumbnail((860, 520), Image.LANCZOS)
        photo = ImageTk.PhotoImage(im)
        canvas.config(image=photo)
        canvas.image = photo
        info_lbl.config(text=f"[{idx[0]+1}/{len(images)}]  {img_path.name}")
        status_lbl.config(text=f"Moved: {moved[0]}")

    def assign(cls):
        img_path = images[idx[0]]
        dest = RAW_BUCKET / cls
        dest.mkdir(parents=True, exist_ok=True)
        shutil.move(str(img_path), str(dest / img_path.name))
        moved[0] += 1
        with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [img_path.name, cls, "1.0", "manual_review", "human"]
            )
        images.pop(idx[0])
        if idx[0] >= len(images):
            idx[0] = max(0, len(images) - 1)
        if images:
            show()
        else:
            info_lbl.config(text="All reviewed!")
            canvas.config(image="")
            root.after(1200, root.destroy)

    def on_key(event):
        k = event.keysym.lower()
        if k == "q":
            root.destroy()
        elif k == "s":
            idx[0] = (idx[0] + 1) % len(images)
            show()
        elif event.char in shortcuts:
            assign(shortcuts[event.char])

    root.bind("<Key>", on_key)
    show()
    root.mainloop()
    print(f"Manual review done. Moved {moved[0]} images.")


# ── HELPERS ───────────────────────────────────────────────────────────────────
def load_existing_log() -> set:
    if not LOG_FILE.exists():
        return set()
    done = set()
    with open(LOG_FILE, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            done.add(row["filename"])
    return done


def copy_to_bucket(img_path: Path, cls: str):
    dest_dir  = RAW_BUCKET / cls
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / f"mdzillur__{img_path.name}"
    if not dest_file.exists():
        shutil.copy2(img_path, dest_file)


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Classify mdzillur flat dataset — no API, no billing"
    )
    parser.add_argument("--no-clip",  action="store_true",
                        help="Use filename patterns only, skip CLIP")
    parser.add_argument("--review",   action="store_true",
                        help="Open Tkinter viewer for _unclassified images")
    parser.add_argument("--sample",   type=int, default=0,
                        help="Only process first N images (testing)")
    parser.add_argument("--resume",   action="store_true",
                        help="Skip images already in the log")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Count images only, no file operations")
    args = parser.parse_args()

    if args.review:
        run_manual_review()
        return

    if not MDZILLUR_DIR.exists():
        print(f"ERROR: {MDZILLUR_DIR} not found")
        sys.exit(1)

    all_images = sorted([
        p for p in MDZILLUR_DIR.iterdir() if p.suffix in IMG_EXT
    ])
    if args.sample:
        all_images = all_images[:args.sample]

    print("MDZILLUR CLASSIFIER  (no-API mode)")
    print("="*60)
    print(f"Source : {MDZILLUR_DIR}")
    print(f"Images : {len(all_images)}")
    print(f"Output : {RAW_BUCKET}")
    print(f"Log    : {LOG_FILE}")
    print()

    if args.dry_run:
        # Quick stats: how many filenames would match?
        matched = sum(
            1 for p in all_images
            if classify_by_filename(p.name)[0] != "_unclassified"
        )
        print(f"[DRY RUN] Filename pattern would match {matched}/{len(all_images)} images.")
        unmatched = len(all_images) - matched
        print(f"          {unmatched} would go to CLIP / _unclassified.")
        return

    already_done = load_existing_log() if args.resume else set()
    to_process   = [p for p in all_images if p.name not in already_done]

    if args.resume:
        print(f"Resuming: {len(already_done)} done, {len(to_process)} remaining\n")

    # Try loading CLIP
    use_clip = (not args.no_clip) and load_clip()
    if not use_clip and not args.no_clip:
        print("  CLIP not installed — using filename patterns only.")
        print("  To enable CLIP:  pip install torch open_clip_torch")
        print()

    # CSV log
    log_exists = LOG_FILE.exists()
    log_fh     = open(LOG_FILE, "a", newline="", encoding="utf-8")
    log_writer = csv.writer(log_fh)
    if not log_exists:
        log_writer.writerow(["filename", "class", "confidence", "method", "notes"])

    class_counts = {c: 0 for c in VALID_CLASSES}
    fn_hits = clip_hits = 0

    # Progress bar (tqdm optional)
    try:
        from tqdm import tqdm
        iterator = tqdm(to_process, unit="img", ncols=80)
    except ImportError:
        iterator = to_process

    for img_path in iterator:
        # Stage 1: filename
        cls, method = classify_by_filename(img_path.name)
        confidence  = "1.0" if cls != "_unclassified" else "0.0"

        if cls != "_unclassified":
            fn_hits += 1
        elif use_clip:
            # Stage 2: CLIP
            try:
                cls, conf_score, method = classify_by_clip(img_path)
                confidence = f"{conf_score:.3f}"
                if cls != "_unclassified":
                    clip_hits += 1
            except Exception as e:
                method     = f"clip_error"
                confidence = "0.0"

        copy_to_bucket(img_path, cls)
        class_counts[cls] = class_counts.get(cls, 0) + 1
        log_writer.writerow([img_path.name, cls, confidence, method, ""])

        if hasattr(iterator, "set_postfix"):
            iterator.set_postfix(cls=cls[:10])
        else:
            done = sum(class_counts.values())
            if done % 200 == 0:
                print(f"  {done}/{len(to_process)} processed...")

    log_fh.flush()
    log_fh.close()

    # Summary
    total        = sum(class_counts.values())
    unclassified = class_counts.get("_unclassified", 0)

    print("\n" + "="*60)
    print("CLASSIFICATION SUMMARY")
    print("="*60)
    print(f"  Filename pattern matched : {fn_hits}")
    print(f"  CLIP zero-shot matched   : {clip_hits}")
    print(f"  Unclassified             : {unclassified}")
    print()

    for cls in VALID_CLASSES:
        n = class_counts.get(cls, 0)
        if n == 0:
            continue
        bar = "█" * min(n // 10, 40)
        print(f"  {cls:<18} {n:>5}  {bar}")

    print(f"\n  Total processed: {total}")

    if unclassified > 0:
        print(f"\n  {unclassified} images need manual review.")
        print("  Run:  python classify_mdzillur.py --review")
        print(f"  Or browse: {RAW_BUCKET / '_unclassified'}")

    print("\nNext step: python organise_fixed.py")


if __name__ == "__main__":
    main()
    