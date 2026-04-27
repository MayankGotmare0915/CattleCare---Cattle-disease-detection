"""
scrape_diseases.py
==================
Downloads images for missing cattle disease classes using icrawler.
Targets: Mastitis, BRD (Bovine Respiratory Disease), Mange

OUTPUT STRUCTURE:
  bucket_A_raw/
    Mastitis/       ← from web scrape
    BRD/            ← from web scrape
    Mange/          ← from web scrape

INSTALL DEPENDENCIES first:
  pip install icrawler Pillow

USAGE:
  python scrape_diseases.py
  python scrape_diseases.py --class Mastitis --count 500
  python scrape_diseases.py --dry-run
"""

import argparse
import hashlib
import os
import shutil
import sys
import time
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE = Path("C:/Projects/CattleCare/cattlecare_dataset")
RAW_BUCKET = BASE / "bucket_A_raw"

# Target image counts per class per search engine
# (Actual downloads may be less — icrawler skips dupes + bad URLs)
TARGET_PER_QUERY = 150   # images per search query
MIN_DIMENSION    = 100   # skip images smaller than 100×100 px

# ── SEARCH QUERIES PER CLASS ──────────────────────────────────────────────────
# Multiple queries per class = more diversity, fewer duplicates.
# Keep queries specific to CATTLE — avoid human medical images.
DISEASE_QUERIES = {

    "Mastitis": [
        "cow mastitis udder disease",
        "bovine mastitis teat inflammation",
        "dairy cow mastitis swollen udder",
        "cattle mastitis milk infection",
        "mastitis cow udder veterinary",
        "heifer mastitis teat lesion",
        "cow udder mastitis clinical signs",
    ],

    "BRD": [
        "bovine respiratory disease cattle",
        "BRD sick cow nasal discharge",
        "cattle pneumonia respiratory bovine",
        "bovine respiratory disease calf symptoms",
        "feedlot cattle respiratory illness",
        "shipping fever bovine cattle",
        "cattle BRD runny nose eye discharge",
    ],

    "Mange": [
        "cattle mange skin disease",
        "bovine mange sarcoptic cow",
        "cattle chorioptic mange lesion",
        "cow mange crusty skin hair loss",
        "bovine psoroptic mange",
        "cattle skin mite mange acarid",
        "mange cow veterinary dermatology",
    ],
}

# ── DEDUP HELPER ──────────────────────────────────────────────────────────────
def file_hash(path: Path) -> str:
    """MD5 of first 64KB — fast enough for dedup."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read(65536))
    return h.hexdigest()


def remove_duplicates(folder: Path) -> int:
    """Remove exact duplicate images within a folder. Returns count removed."""
    seen = {}
    removed = 0
    for img in folder.iterdir():
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        digest = file_hash(img)
        if digest in seen:
            img.unlink()
            removed += 1
        else:
            seen[digest] = img
    return removed


def filter_small_images(folder: Path, min_dim: int) -> int:
    """Remove images smaller than min_dim in either dimension."""
    try:
        from PIL import Image
    except ImportError:
        print("  Pillow not installed — skipping size filter (pip install Pillow)")
        return 0

    removed = 0
    for img_path in folder.iterdir():
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        try:
            with Image.open(img_path) as im:
                w, h = im.size
            if w < min_dim or h < min_dim:
                img_path.unlink()
                removed += 1
        except Exception:
            img_path.unlink()   # corrupt / unreadable
            removed += 1
    return removed


# ── CRAWL ONE QUERY ───────────────────────────────────────────────────────────
def crawl_query(query: str, dest_folder: Path, count: int, engine: str = "bing"):
    """
    Download `count` images matching `query` into a temp subfolder,
    then merge into dest_folder with unique filenames.
    """
    try:
        from icrawler.builtin import BingImageCrawler, GoogleImageCrawler
    except ImportError:
        print("ERROR: icrawler not installed. Run:  pip install icrawler")
        sys.exit(1)

    # Temp folder for this query's downloads
    safe_q    = query.replace(" ", "_").replace("/", "-")[:60]
    temp_dir  = dest_folder / f"_tmp_{safe_q}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        if engine == "google":
            crawler = GoogleImageCrawler(
                storage={"root_dir": str(temp_dir)},
                log_level=50,       # ERROR only — suppress INFO spam
            )
        else:
            crawler = BingImageCrawler(
                storage={"root_dir": str(temp_dir)},
                log_level=50,
            )

        crawler.crawl(
            keyword=query,
            max_num=count,
            file_idx_offset="auto",
            filters={"type": "photo"},  # skip clipart/illustrations
        )
    except Exception as e:
        print(f"    Crawler error ({engine}): {e}")

    # Merge temp → dest with unique names
    moved = 0
    prefix = safe_q[:40]
    for img in temp_dir.iterdir():
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
            continue
        # Convert .webp → .jpg on the fly (optional — just rename)
        new_name   = f"{prefix}__{img.name}"
        dest_file  = dest_folder / new_name
        if not dest_file.exists():
            shutil.move(str(img), str(dest_file))
            moved += 1
        else:
            img.unlink()

    # Clean up temp dir
    try:
        shutil.rmtree(temp_dir)
    except Exception:
        pass

    return moved


# ── MAIN ──────────────────────────────────────────────────────────────────────
def scrape_class(class_name: str, queries: list, target_per_query: int, dry_run: bool):
    dest = RAW_BUCKET / class_name
    dest.mkdir(parents=True, exist_ok=True)

    existing = len(list(dest.glob("*.[jJpP][pPnN][gGeE]*")))
    print(f"\n{'='*60}")
    print(f"  Class   : {class_name}")
    print(f"  Queries : {len(queries)}")
    print(f"  Target  : ~{len(queries) * target_per_query} images")
    print(f"  Existing: {existing} images already in bucket")
    print(f"  Dest    : {dest}")

    if dry_run:
        print("  [DRY RUN] — no downloads performed")
        for q in queries:
            print(f"    Would crawl: {q!r}")
        return

    total_downloaded = 0
    for i, query in enumerate(queries, 1):
        print(f"\n  [{i}/{len(queries)}] Crawling: {query!r}")

        # Try Bing first, fall back to Google
        for engine in ("bing", "google"):
            n = crawl_query(query, dest, target_per_query, engine=engine)
            print(f"    {engine:6s}: +{n} images")
            total_downloaded += n
            if n > 0:
                break               # got something from first engine
            time.sleep(1)           # brief pause before fallback

        time.sleep(2)               # be polite between queries

    # Post-processing
    print(f"\n  Post-processing {class_name}...")
    dup_removed  = remove_duplicates(dest)
    small_removed = filter_small_images(dest, MIN_DIMENSION)
    final_count  = len(list(dest.glob("*.[jJpP][pPnN][gGeE]*")))

    print(f"  Duplicates removed : {dup_removed}")
    print(f"  Tiny images removed: {small_removed}")
    print(f"  Final image count  : {final_count}")

    return final_count


def main():
    parser = argparse.ArgumentParser(description="Scrape cattle disease images")
    parser.add_argument(
        "--class",
        dest="cls",
        choices=list(DISEASE_QUERIES.keys()) + ["all"],
        default="all",
        help="Which class to scrape (default: all missing classes)"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=TARGET_PER_QUERY,
        help=f"Images per search query (default: {TARGET_PER_QUERY})"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without downloading"
    )
    args = parser.parse_args()

    print("CATTLE DISEASE IMAGE SCRAPER")
    print("="*60)
    print(f"Output bucket : {RAW_BUCKET}")
    print(f"Min dimension : {MIN_DIMENSION}px")
    print(f"Target/query  : {args.count}")
    print()

    classes_to_scrape = (
        DISEASE_QUERIES
        if args.cls == "all"
        else {args.cls: DISEASE_QUERIES[args.cls]}
    )

    totals = {}
    for class_name, queries in classes_to_scrape.items():
        count = scrape_class(class_name, queries, args.count, args.dry_run)
        totals[class_name] = count or 0

    # Summary
    print("\n" + "="*60)
    print("SCRAPING SUMMARY")
    print("="*60)
    for cls, n in totals.items():
        bar = "█" * min(n // 20, 40)
        print(f"  {cls:<12} {n:>5} images  {bar}")

    print("\nNext steps:")
    print("  1. Visually review scraped images — remove any non-cattle images")
    print("     (web scrapes always have some noise)")
    print("  2. python classify_mdzillur.py   (auto-label the flat mdzillur set)")
    print("  3. python organise_fixed.py       (run updated organiser)")
    print("  4. python count_and_plan.py       (check class balance)")


if __name__ == "__main__":
    main()