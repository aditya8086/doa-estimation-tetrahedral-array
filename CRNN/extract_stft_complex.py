#!/usr/bin/env python3
import os
import shutil
import numpy as np
from scipy.signal import stft
from tqdm import tqdm

input_dir            = r"C:\Users\csio\doa_project\dataset"            
complex_features_dir = r"C:\Users\csio\doa_project\features_complex" 

if os.path.isdir(complex_features_dir):
    shutil.rmtree(complex_features_dir)
os.makedirs(complex_features_dir, exist_ok=True)

fs       = 20000
nperseg  = 1024     # ≈ 51 ms window
noverlap = 512      # 50% overlap

files = [f for f in os.listdir(input_dir) if f.lower().endswith(".npy")]
print(f"Complex STFT → {len(files)} files | fs={fs}, nperseg={nperseg}, noverlap={noverlap}\n")

for fname in tqdm(files, desc="Files", unit="file"):
    data = np.load(os.path.join(input_dir, fname)) 
    stft_ch = []

    for ch in range(4):
        _, _, Zxx = stft(
            data[ch],
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            boundary=None
        )
        C = np.stack([Zxx.real, Zxx.imag], axis=-1).astype(np.float32)
        stft_ch.append(C)

    feat = np.stack(stft_ch, axis=0)
    out_path = os.path.join(complex_features_dir, fname)
    np.save(out_path, feat)

print(f"\nAll complex STFT features saved to: {complex_features_dir}")
