from roboflow import Roboflow
import os
import shutil
from pathlib import Path

# ── PASTE YOUR API KEY HERE ──────────────────────────────────────
API_KEY = "WPGJ1AQ6EWqaOAS59vq8"

rf = Roboflow(api_key=API_KEY)

# Where to save everything
SAVE_DIR = "roboflow_downloads"
os.makedirs(SAVE_DIR, exist_ok=True)

# ── LIST OF DATASETS TO DOWNLOAD ────────────────────────────────
# Format: (workspace_name, project_name, version_number, your_folder_name)
# These are the real dataset addresses on Roboflow

datasets_to_download = [
    # Mastitis dataset — 3,969 images
    ("kirubel-yemane",
     "cow-and-mastitis-detection-ufyb8",
     1,
     "kirubel_mastitis"),

    # Multi-cattle diseases — 3,294 images
    ("mdzillur-rahaman-rohan",
     "cattle_disease-detection",
     1,
     "mdzillur_cattle"),

    # SLIIT cattle diseases — 834 images
    ("sliit-kuemd",
     "cattle-diseases",
     2,
     "sliit_cattle"),

    # LSD detection — 1,019 images
    ("qq-mgfrz",
     "lumpy-skin-disease-detection",
     1,
     "qq_lsd"),

    # FMD — 99 images
    ("moaz",
     "fmd-detection",
     1,
     "moaz_fmd"),
]

print("Starting downloads...\n")

for workspace, project, version, folder_name in datasets_to_download:
    print(f"Downloading: {folder_name}")
    print(f"  From: {workspace}/{project} v{version}")

    try:
        proj    = rf.workspace(workspace).project(project)
        dataset = proj.version(version).download(
            "folder",                         # ← THIS IS THE KEY
                                              # "folder" = classification format
                                              # images sorted into class folders
            location=f"{SAVE_DIR}/{folder_name}",
            overwrite=True
        )
        print(f"  ✓ Saved to {SAVE_DIR}/{folder_name}")

    except Exception as e:
        print(f"  ✗ Failed: {e}")
        print(f"  → Try searching '{project}' on universe.roboflow.com")
        print(f"    and download manually as 'Folder' format")

    print()

print("All downloads attempted.")
print(f"Check the '{SAVE_DIR}' folder.")