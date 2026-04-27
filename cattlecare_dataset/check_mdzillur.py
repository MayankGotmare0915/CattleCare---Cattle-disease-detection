# check_mdzillur.py
from pathlib import Path

folder = Path("roboflow_downloads/mdzillur_cattle/train")

print("Contents of mdzillur_cattle/train/:")
print()

if not folder.exists():
    print("Folder not found!")
else:
    items = list(folder.iterdir())
    dirs  = [i for i in items if i.is_dir()]
    imgs  = [i for i in items
             if i.suffix.lower() in
             ['.jpg','.jpeg','.png']]

    print(f"Subfolders ({len(dirs)}):")
    for d in sorted(dirs):
        img_count = len(list(d.glob("*.jpg"))) + \
                    len(list(d.glob("*.png")))
        print(f"  {d.name}  →  {img_count} images")

    print(f"\nFlat images in root: {len(imgs)}")

    if imgs:
        print("\nSample filenames (first 10):")
        for img in imgs[:10]:
            print(f"  {img.name}")
        print()
        print("ACTION: Look at these images visually")
        print("and decide what disease class they show.")