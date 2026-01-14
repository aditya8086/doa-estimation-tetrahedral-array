#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import random

# — ADJUST THIS to point at your .npy stacks folder — 
dataset_dir = r"C:\Users\csio\doa_project\dataset"  

# where to dump your PNGs
plots_dir = os.path.join(dataset_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

# STFT parameter candidates to compare (∆f, hop)
params = [
    (64,  32),   # Δf≈62.5 Hz, hop=16 ms
    (128, 64),   # Δf≈31 Hz,   hop=32 ms
    (256, 128),  # Δf≈15.6 Hz, hop=64 ms
    (512, 256),  # Δf≈7.8 Hz,  hop=128 ms
]

eps = 1e-8   # to avoid log(0)

# list all .npy stacks
files = [f for f in os.listdir(dataset_dir) if f.lower().endswith(".npy")]
if not files:
    raise RuntimeError(f"No .npy files found in {dataset_dir!r}")

# show numbered list
print(f"Found {len(files)} feature files.\n")
for idx, fn in enumerate(files):
    print(f"  [{idx:3d}] {fn}")
print()

# prompt user
choice = input("Enter an index (or full filename) from above (leave blank for random): ").strip()
if choice == "":
    choice = random.choice(files)
elif choice.isdigit():
    i = int(choice)
    if 0 <= i < len(files):
        choice = files[i]
    else:
        print(f"Index {i} out of range, using random.")
        choice = random.choice(files)
elif choice not in files:
    print(f"Filename {choice!r} not found, using random.")
    choice = random.choice(files)

print("→ Visualizing:", choice)

# load only the first mic channel
full_path = os.path.join(dataset_dir, choice)
data      = np.load(full_path)   # shape (4, N)
signal    = data[0]              # channel 0

fs = 4000  # sampling rate

# loop through your STFT settings
for nperseg, noverlap in params:
    f, t, Zxx = stft(
        signal, fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        boundary=None
    )
    mag    = np.abs(Zxx)
    mag_db = 10.0 * np.log10(mag + eps)

    # plot side by side
    plt.figure(figsize=(12, 4))

    # raw magnitude
    plt.subplot(1, 2, 1)
    plt.pcolormesh(t, f, mag, shading='gouraud')
    plt.title(f"Mag — n={nperseg}, o={noverlap}")
    plt.xlabel("Time [s]"); plt.ylabel("Freq [Hz]")
    plt.colorbar(label='Amplitude')

    # dB-scaled
    plt.subplot(1, 2, 2)
    plt.pcolormesh(t, f, mag_db, shading='gouraud')
    plt.title(f"dB — n={nperseg}, o={noverlap}")
    plt.xlabel("Time [s]"); plt.ylabel("Freq [Hz]")
    plt.colorbar(label='dB')

    plt.tight_layout()
    out_name = f"{os.path.splitext(choice)[0]}_n{nperseg}_o{noverlap}.png"
    out_path = os.path.join(plots_dir, out_name)
    plt.savefig(out_path, dpi=300)
    plt.close()

print("\nDone! Plots (mag vs. dB) saved to:", plots_dir)