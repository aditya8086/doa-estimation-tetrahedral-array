#!/usr/bin/env python3
import os
import re
import shutil
import numpy as np
import soundfile as sf
from collections import defaultdict
from tqdm import tqdm

dataset_dir   = r"C:\Users\csio\doa_project\dataset"
chunks_folder = r"C:\Users\csio\doa_project\chunks"

os.makedirs(dataset_dir, exist_ok=True)

required_mics = ("base1","base2","base3","top")

# DP (DOA) regex & grouping
pat_dp    = re.compile(
    r"^dp(0|30|60)"            # elevation
    r"([1-9]|1[0-2])_"         # point 1–12
    r"(base1|base2|base3|top)" # mic
    r"_chunk(\d+)\.wav$"       # index
)
groups_dp = defaultdict(dict)

for fn in os.listdir(chunks_folder):
    m = pat_dp.match(fn)
    if not m: continue
    elev, point, mic, idx = m.groups()
    groups_dp[(int(elev), int(point), idx)][mic] = fn

# Noise regex & grouping 
pat_noise    = re.compile(r"^Noise_(base1|base2|base3|top)_chunk(\d+)\.wav$")
groups_noise = defaultdict(dict)

for fn in os.listdir(chunks_folder):
    m = pat_noise.match(fn)
    if not m: continue
    mic, idx = m.groups()  
    groups_noise[idx][mic] = fn

# Stack & save both DP and Noise into the same folder —

saved_dp = skipped_dp = 0
for (elev, point, idx), files in tqdm(list(groups_dp.items()), desc="Stacking DOA", unit="item"):
    out_name = f"dp{elev}{point}_chunk{idx.zfill(4)}.npy"
    out_path = os.path.join(dataset_dir, out_name)
    if os.path.exists(out_path):
        continue

    if all(m in files for m in required_mics):
        arr = [sf.read(os.path.join(chunks_folder, files[m]))[0] for m in required_mics]
        np.save(out_path, np.stack(arr, axis=0))
        saved_dp += 1
    else:
        skipped_dp += 1

saved_n = skipped_n = 0
for idx, files in tqdm(list(groups_noise.items()), desc="Stacking Noise", unit="item"):
    out_name = f"Noise_chunk{idx.zfill(4)}.npy"
    out_path = os.path.join(dataset_dir, out_name)
    if os.path.exists(out_path):
        continue

    if all(m in files for m in required_mics):
        arr = [sf.read(os.path.join(chunks_folder, files[m]))[0] for m in required_mics]
        np.save(out_path, np.stack(arr, axis=0))
        saved_n += 1
    else:
        skipped_n += 1

print(f"\nDOA stacks saved:   {saved_dp}, skipped (missing mics): {skipped_dp}")
print(f"Noise stacks saved: {saved_n}, skipped (missing mics): {skipped_n}")
print("All .npy files are now in:", dataset_dir)
