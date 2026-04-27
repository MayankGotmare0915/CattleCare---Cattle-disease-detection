# separate_sources.py
# Moves images into two buckets:
#   bucket_A_raw      = real unaugmented images
#   bucket_B_roboflow = already augmented images from Roboflow

import shutil
from pathlib import Path

# These are the Roboflow dataset folder names you downloaded
ROBOFLOW_SOURCES = [
    "kirubel_mastitis",
    "mdzillur_cattle",
    "sliit_cattle",
    "qq_lsd",
    "moaz_fmd",
]

CLASSES = [
    "LSD", "Mastitis", "FMD", "BRD", "Mange",
    "Healthy_Cow", "Healthy_Buffalo",
]

for cls in CLASSES:
    raw_dir = Path("raw_images") / cls
    if not raw_dir.exists():
        continue

    bucket_a = Path("bucket_A_raw") / cls
    bucket_b = Path("bucket_B_roboflow") / cls
    bucket_a.mkdir(parents=True, exist_ok=True)
    bucket_b.mkdir(parents=True, exist_ok=True)

    for img in raw_dir.glob("*.*"):
        # Check if this image came from Roboflow
        is_roboflow = any(
            src in img.name for src in ROBOFLOW_SOURCES
        )

        if is_roboflow:
            shutil.copy2(img, bucket_b / img.name)
        else:
            shutil.copy2(img, bucket_a / img.name)

    a_count = len(list(bucket_a.glob("*.*")))
    b_count = len(list(bucket_b.glob("*.*")))
    print(f"{cls}: {a_count} raw | {b_count} Roboflow")