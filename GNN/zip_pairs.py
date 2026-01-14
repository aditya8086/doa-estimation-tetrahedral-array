import zipfile
import os
from tqdm import tqdm

pairs_dir = r"C:\Users\csio\projects\new_gnn_project\pairs"
zip_path  = r"C:\Users\csio\projects\new_gnn_project\pairs.zip"

# Collect all file paths first
all_files = []
for root, _, files in os.walk(pairs_dir):
    for file in files:
        full_path = os.path.join(root, file)
        rel_path = os.path.relpath(full_path, pairs_dir)
        all_files.append((full_path, os.path.join("pairs", rel_path)))

# Zip with progress bar
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for full_path, rel_path in tqdm(all_files, desc="Zipping pairs"):
        zipf.write(full_path, arcname=rel_path)

print(f"\nZipped {len(all_files)} files to: {zip_path}")
