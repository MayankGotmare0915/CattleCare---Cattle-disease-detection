# diagnose.py  — run this RIGHT NOW
# It tells you exactly what got downloaded and where

import os
from pathlib import Path

def scan_folder(base_path, max_depth=4):
    base = Path(base_path)
    if not base.exists():
        print(f"  FOLDER DOES NOT EXIST: {base_path}")
        return

    print(f"\nScanning: {base_path}")
    print("-" * 50)

    found_images = 0
    found_folders = 0

    for root, dirs, files in os.walk(base):
        # Calculate depth
        depth = len(Path(root).relative_to(base).parts)
        if depth > max_depth:
            continue

        indent = "  " * depth
        folder_name = Path(root).name

        # Count images in this folder
        images = [f for f in files
                  if f.lower().endswith(
                      ('.jpg', '.jpeg', '.png', '.bmp'))]

        if images:
            print(f"{indent}{folder_name}/ "
                  f"→ {len(images)} images")
            found_images += len(images)
            found_folders += 1
        elif depth <= 2:
            print(f"{indent}{folder_name}/  (no images)")

    print(f"\n  Total: {found_images} images "
          f"in {found_folders} folders")

# Scan everything
for folder in ["roboflow_downloads", "kaggle_downloads",
               "raw_images", "bucket_A_raw", "bucket_B_roboflow"]:
    scan_folder(folder)