import os
from PIL import Image

BASE = os.path.join("dataset", "data")
SPLITS = {
    "train": os.path.join(BASE, "training_set", "training_set"),
    "test": os.path.join(BASE, "test_set", "test_set"),
}

for split, path in SPLITS.items():
    print(f"\n=== {split.upper()} ({path}) ===")
    if not os.path.isdir(path):
        print("  Folder not found, check the path")
        continue
    for cls in sorted(os.listdir(path)):
        cls_path = os.path.join(path, cls)
        if not os.path.isdir(cls_path):
            continue
        files = os.listdir(cls_path)
        bad = []
        for f in files:
            try:
                with Image.open(os.path.join(cls_path, f)) as img:
                    img.verify()
            except Exception:
                bad.append(f)
        print(f"  {cls}: {len(files)} files, {len(bad)} unreadable")
        for b in bad:
            print(f"     bad file: {b}")