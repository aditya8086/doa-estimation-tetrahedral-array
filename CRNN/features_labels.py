#!/usr/bin/env python3
import os
import re
import pandas as pd

feat_dir   = r"C:\Users\csio\doa_project\features"
output_csv = r"C:\Users\csio\doa_project\labels.csv"

all_files = [f for f in os.listdir(feat_dir) if f.lower().endswith(".npy")]
print(f"Found {len(all_files)} feature files in {feat_dir}")

rows = []

dp_pat = re.compile(r"^dp(0|30|60)([1-9]|1[0-2])_chunk(\d{4})\.npy$")
for fn in all_files:
    m = dp_pat.match(fn)
    if m:
        elev_str, pt_str, idx = m.groups()
        elevation = int(elev_str)
        point     = int(pt_str)            # 1..12
        azimuth   = (point - 1) * 30       # 0, 30, …,330
        rows.append({
            "filename":  fn,
            "azimuth":   azimuth,
            "elevation": elevation
        })
        continue

    # label the noise files
    if fn.startswith("Noise_chunk") and fn.endswith(".npy"):
        rows.append({
            "filename":  fn,
            "azimuth":   -1,
            "elevation": -1
        })
        continue
    raise RuntimeError(f"Unrecognized file name: {fn}")

df = pd.DataFrame(rows)
df.to_csv(output_csv, index=False)
print(f"Wrote {len(df)} rows to {output_csv}")

if len(df) != len(all_files):
    print(f"Warning: row count ({len(df)}) != file count ({len(all_files)})")
