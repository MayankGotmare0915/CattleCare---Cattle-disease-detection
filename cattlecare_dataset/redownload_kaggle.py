# redownload_kaggle.py
import subprocess
from pathlib import Path

# These two failed — redownload with explicit path
failed = [
    ("saurabhshahane/lumpy-skin-disease-dataset",
     "kaggle_downloads/lumpy-skin-disease-dataset"),

    ("maryam18/animal-disease-classification",
     "kaggle_downloads/animal-disease-classification"),
]

for dataset, save_path in failed:
    Path(save_path).mkdir(parents=True, exist_ok=True)
    print(f"Downloading: {dataset}")

    result = subprocess.run([
        "kaggle", "datasets", "download",
        "-d", dataset,
        "-p", save_path,
        "--unzip",
        "--force",   # redownload even if exists
    ], capture_output=True, text=True)

    print(result.stdout)
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")

    # Check what's there now
    all_imgs = list(Path(save_path).rglob("*.jpg")) + \
               list(Path(save_path).rglob("*.png"))
    print(f"Found {len(all_imgs)} images after download\n")