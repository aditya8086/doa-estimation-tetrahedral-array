#!/usr/bin/env python3
import os
import shutil
import numpy as np
from scipy.signal import stft
from tqdm import tqdm

# — PARAMETERS —
input_dir            = r"C:\Users\csio\doa_project\dataset"            # 4×N raw stacks
complex_features_dir = r"C:\Users\csio\doa_project\features_complex"  # output: real+imag STFTs

# 0) Clear out any old complex features
if os.path.isdir(complex_features_dir):
    shutil.rmtree(complex_features_dir)
os.makedirs(complex_features_dir, exist_ok=True)

# 1) STFT settings — for fs = 20 kHz
fs       = 20000
nperseg  = 1024     # ≈ 51 ms window
noverlap = 512      # 50% overlap

# 2) List all stacked chunks
files = [f for f in os.listdir(input_dir) if f.lower().endswith(".npy")]
print(f"Complex STFT → {len(files)} files | fs={fs}, nperseg={nperseg}, noverlap={noverlap}\n")

# 3) Compute & save STFTs
for fname in tqdm(files, desc="Files", unit="file"):
    data = np.load(os.path.join(input_dir, fname))  # shape: (4, N) = (4, 5000)
    stft_ch = []

    for ch in range(4):
        _, _, Zxx = stft(
            data[ch],
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            boundary=None
        )
        # Real + Imaginary stacking: (F, T, 2)
        C = np.stack([Zxx.real, Zxx.imag], axis=-1).astype(np.float32)
        stft_ch.append(C)

    # Final shape: (4, F, T, 2)
    feat = np.stack(stft_ch, axis=0)
    out_path = os.path.join(complex_features_dir, fname)
    np.save(out_path, feat)

print(f"\nAll complex STFT features saved to: {complex_features_dir}")
