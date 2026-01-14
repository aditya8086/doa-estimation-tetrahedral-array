#!/usr/bin/env python3
import os
import numpy as np
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt

feat_dir   = r"C:\Users\csio\doa_project\features_complex"  
output_dir = r"C:\Users\csio\doa_project\plots_features_complex"  

os.makedirs(output_dir, exist_ok=True)

shapes = Counter()
total_files = 0

for fname in tqdm(os.listdir(feat_dir), desc="Scanning features"):
    if not fname.lower().endswith(".npy"):
        continue
    total_files += 1
    arr = np.load(os.path.join(feat_dir, fname))
    shapes[arr.shape] += 1

print(f"\nProcessed {total_files} feature files.\n")
print("Unique shapes and their counts:")
for shape, cnt in shapes.items():
    print(f"  {shape}: {cnt}")

print("\nExtracted frequency/time dims from (4,F,T) shapes:")
for shape in shapes:
    if len(shape) == 3 and shape[0] == 4:
        _, F, T = shape
        print(f"  Found (4, {F}, {T})  →  F = {F},  T = {T}")

files = sorted(f for f in os.listdir(feat_dir) if f.lower().endswith(".npy"))
if not files:
    raise RuntimeError(f"No .npy files found in {feat_dir!r}")

sample_fname = files[0]
sample_path  = os.path.join(feat_dir, sample_fname)
tensor       = np.load(sample_path)

if tensor.ndim == 4 and tensor.shape[0] == 4 and tensor.shape[-1] == 2:
    # Complex features: (4, F, T, 2)
    is_complex = True
    F, T = tensor.shape[1], tensor.shape[2]
elif tensor.ndim == 3 and tensor.shape[0] == 4:
    # Magnitude only features: (4, F, T)
    is_complex = False
    F, T = tensor.shape[1], tensor.shape[2]
else:
    raise RuntimeError(f"Unexpected tensor shape for {sample_fname}: {tensor.shape}")

print(f"\nVisualizing sample: {sample_fname}  →  shape = {tensor.shape}")

# Create a 4×3 grid (Real / Imag / Mag) if complex, or 4×1 (just Mag) if magnitude only
if is_complex:
    fig, axes = plt.subplots(4, 3, figsize=(18, 12))
    for mic in range(4):
        real_part      = tensor[mic, :, :, 0]
        imag_part      = tensor[mic, :, :, 1]
        magnitude_part = np.sqrt(real_part**2 + imag_part**2)

        axes[mic, 0].imshow(real_part,      aspect='auto', origin='lower', cmap='RdBu_r')
        axes[mic, 0].set_title(f"Mic {mic+1} — Real")
        axes[mic, 0].set_ylabel("Freq Bins")
        axes[mic, 0].set_xlabel("Time Frames")

        axes[mic, 1].imshow(imag_part,      aspect='auto', origin='lower', cmap='RdBu_r')
        axes[mic, 1].set_title(f"Mic {mic+1} — Imag")
        axes[mic, 1].set_xlabel("Time Frames")

        im = axes[mic, 2].imshow(magnitude_part, aspect='auto', origin='lower', cmap='viridis')
        axes[mic, 2].set_title(f"Mic {mic+1} — Magnitude")
        axes[mic, 2].set_xlabel("Time Frames")

    cbar = fig.colorbar(im, ax=axes[:,2].reshape(-1,1), fraction=0.015, pad=0.04)
    cbar.set_label("Magnitude")
    plt.suptitle(f"STFT (complex) Features → {sample_fname}", fontsize=16)
    plt.tight_layout(rect=[0,0,1,0.96])

else:
    fig, axes = plt.subplots(4, 1, figsize=(8, 12))
    for mic in range(4):
        mag_only = tensor[mic, :, :]  # shape (F, T)
        im = axes[mic].imshow(mag_only, aspect='auto', origin='lower', cmap='viridis')
        axes[mic].set_title(f"Mic {mic+1} — Magnitude")
        axes[mic].set_ylabel("Freq Bins")
        axes[mic].set_xlabel("Time Frames")
    cbar = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.04)
    cbar.set_label("Magnitude")
    plt.suptitle(f"STFT (magnitude only) → {sample_fname}", fontsize=16)
    plt.tight_layout(rect=[0,0,1,0.96])

out_png = os.path.join(output_dir, sample_fname.replace(".npy", ".png"))
plt.savefig(out_png, dpi=300)
plt.close()
print(f"Saved visualization to: {out_png}")
