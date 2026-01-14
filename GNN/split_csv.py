import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

# CONFIG
PAIR_DIR = r"C:\Users\csio\projects\new_gnn_project\pairs"
CSV_DIR = r"C:\Users\csio\projects\new_gnn_project\splits"
os.makedirs(CSV_DIR, exist_ok=True)

# ─── Collect all filenames and labels using strict regex ─────────────
file_rows = []

# Pattern: dp{elevation: 0|30|60}{point: 1-12}_chunkXXXXX_feat_pair.npy
dp_pat = re.compile(r"^dp(0|30|60)([1-9]|1[0-2])_chunk\d+_feat_pair\.npy$")

for fname in os.listdir(PAIR_DIR):
    if not fname.endswith(".npy") or not fname.startswith("dp"):
        continue

    match = dp_pat.match(fname)
    if match:
        elev_str, pt_str = match.groups()
        elevation = int(elev_str)
        point = int(pt_str)  # 1..12
        azimuth = (point - 1) * 30
        file_rows.append((fname, azimuth, elevation))
    else:
        print(f"Skipping unrecognized filename: {fname}")

# Build dataframe
df = pd.DataFrame(file_rows, columns=["filename", "azimuth", "elevation"])

# ─── Stratification Label ─────────────────────────────────────────
df["strat_label"] = df["azimuth"].astype(str) + "_" + df["elevation"].astype(str)

# ─── Stratified Splitting ─────────────────────────────────────────
sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
train_val_idx, test_idx = next(sss1.split(df, df["strat_label"]))

train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
test_df      = df.iloc[test_idx].reset_index(drop=True)

sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.1765, random_state=42)  # ~15% of total
train_idx, val_idx = next(sss2.split(train_val_df, train_val_df["strat_label"]))

train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
val_df   = train_val_df.iloc[val_idx].reset_index(drop=True)

# ─── Save splits ───────────────────────────────────────────────────
train_df.drop(columns=["strat_label"]).to_csv(os.path.join(CSV_DIR, "train.csv"), index=False)
val_df.drop(columns=["strat_label"]).to_csv(os.path.join(CSV_DIR, "val.csv"), index=False)
test_df.drop(columns=["strat_label"]).to_csv(os.path.join(CSV_DIR, "test.csv"), index=False)

print(f"Stratified splits saved to {CSV_DIR}")
print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
