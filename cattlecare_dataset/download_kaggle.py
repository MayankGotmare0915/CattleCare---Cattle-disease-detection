import subprocess
import os
import zipfile
from pathlib import Path

SAVE_DIR = "kaggle_downloads"
os.makedirs(SAVE_DIR, exist_ok=True)

# List of Kaggle dataset addresses
# Find these by going to the dataset page on kaggle.com
# and looking at the URL: kaggle.com/datasets/USERNAME/DATASETNAME

kaggle_datasets = [
    # LSD dataset
    "saurabhshahane/lumpy-skin-disease-dataset",

    # Another LSD dataset
    "shivamagarwal29/cow-lumpy-disease-dataset",

    # Multi-disease cattle
    "devang03mgr/cattle-diseases-datasets",

    # Animal disease classification
    "maryam18/animal-disease-classification",
]

for dataset in kaggle_datasets:
    folder_name = dataset.split("/")[1]  # get the dataset name part
    save_path   = f"{SAVE_DIR}/{folder_name}"
    os.makedirs(save_path, exist_ok=True)

    print(f"Downloading: {dataset}")

    # This command downloads the dataset as a zip file
    result = subprocess.run([
        "kaggle", "datasets", "download",
        "-d", dataset,
        "-p", save_path,
        "--unzip"   # automatically unzip after download
    ], capture_output=True, text=True)

    if result.returncode == 0:
        print(f"  ✓ Saved to {save_path}")
    else:
        print(f"  ✗ Failed: {result.stderr}")

    print()

print("Done. Check the 'kaggle_downloads' folder.")