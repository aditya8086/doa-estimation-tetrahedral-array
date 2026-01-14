#!/usr/bin/env python3
import os
import zipfile
from tqdm import tqdm

def zip_folder(src_dir: str, zip_path: str):
    """
    Zips up the entire src_dir into zip_path, showing a tqdm progress bar.
    """
    # 1) Gather all files
    file_paths = []
    for root, _, files in os.walk(src_dir):
        for f in files:
            full = os.path.join(root, f)
            # store relative path so zip has the same folder structure
            rel  = os.path.relpath(full, start=os.path.dirname(src_dir))
            file_paths.append((full, rel))

    # 2) Write to zip with progress
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for full, rel in tqdm(file_paths, desc="Zipping features", unit="file"):
            zf.write(full, arcname=rel)

if __name__ == "__main__":
    SRC_DIR = r"C:\Users\csio\doa_project\features"
    OUT_ZIP = r"C:\Users\csio\doa_project\features.zip"
    zip_folder(SRC_DIR, OUT_ZIP)
    print(f"Done! Zipped {SRC_DIR} → {OUT_ZIP}")
