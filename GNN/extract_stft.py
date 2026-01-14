# scripts/1_extract_stft.py

import os
import sys
import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from features import stft_frame

CHUNK_DIR = r"C:\Users\csio\projects\new_gnn_project\4ChannelStackedChunk"
FEATURE_DIR = r"C:\Users\csio\projects\new_gnn_project\features"
N_FFT = 2048

os.makedirs(FEATURE_DIR, exist_ok=True)

chunk_files = sorted([f for f in os.listdir(CHUNK_DIR) if f.endswith(".npy")])

print(f"Found {len(chunk_files)} chunk files")

for fname in tqdm(chunk_files, desc="Extracting STFT features"):
    chunk_path = os.path.join(CHUNK_DIR, fname)
    data = np.load(chunk_path)  # shape: [4, 512]

    stfts = [stft_frame(data[i], n_fft=N_FFT) for i in range(4)] 
    Xi_real = np.real(stfts)
    Xi_imag = np.imag(stfts)

    feat = np.stack([Xi_real, Xi_imag], axis=1).astype(np.float32) 

    out_path = os.path.join(FEATURE_DIR, fname.replace(".npy", "_feat.npy"))
    np.save(out_path, feat)

print("\nSTFT feature extraction complete!")
