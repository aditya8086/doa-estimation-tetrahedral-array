# scripts/2_make_pairs.py

import os
import numpy as np
from tqdm import tqdm

# ─── CONFIG ───────────────────────────────────────────────────────────────
FEATURE_DIR = r"C:\Users\csio\projects\new_gnn_project\features"
PAIR_DIR    = r"C:\Users\csio\projects\new_gnn_project\pairs"

mic_coords = [
    (-0.04, -0.02309, 0.0),   # Mic 0
    (+0.04, -0.02309, 0.0),   # Mic 1
    (0.0,   +0.04619, 0.0),   # Mic 2
    (0.0,   0.0,     0.06557) # Mic 3 (top)
]
#centre is at (0,0,0) with r=0.04619meters

os.makedirs(PAIR_DIR, exist_ok=True)

# ─── LOAD AND CONVERT EACH FILE ───────────────────────────────────────────
files = sorted(f for f in os.listdir(FEATURE_DIR) if f.endswith(".npy"))
print(f"Found {len(files)} feature files")

for fname in tqdm(files, desc="Forming 12 pairwise features"):
    path = os.path.join(FEATURE_DIR, fname)
    data = np.load(path)  # shape: [4, 2, F]
    mic_stfts = data
    F = data.shape[2]

    pair_feats = []
    for i in range(4):
        for j in range(4):
            if i == j:
                continue
            Xi = mic_stfts[i]  # [2, F]
            Xj = mic_stfts[j]
            part = np.concatenate([Xi[0], Xi[1], Xj[0], Xj[1]])  # 4F

            ci = np.array(mic_coords[i], dtype=np.float32)
            cj = np.array(mic_coords[j], dtype=np.float32)
            coord = np.concatenate([ci, cj], axis=0)  # [6]

            final_feat = np.concatenate([part.astype(np.float32), coord])  # [4F+6]
            pair_feats.append(final_feat)

    out = np.stack(pair_feats, axis=0)  # shape: [12, 4F+6]
    out_path = os.path.join(PAIR_DIR, fname.replace(".npy", "_pair.npy"))
    np.save(out_path, out)

print("\nPairwise features saved.")
