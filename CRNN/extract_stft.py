import os
import shutil
import numpy as np
from scipy.signal import stft
from tqdm import tqdm

# — PARAMETERS —
input_dir    = r"C:\Users\csio\doa_project\dataset"   # your 4×N stacks
features_dir = r"C:\Users\csio\doa_project\features"  # output folder for dB-scaled features

# 0) Clear out old features
if os.path.isdir(features_dir):
    shutil.rmtree(features_dir)
os.makedirs(features_dir, exist_ok=True)

# 1) Correct STFT settings for fs = 20 kHz
fs       = 20000
nperseg  = 1024
noverlap = 512
eps      = 1e-8

# 2) Build list of .npy audio stacks
files = [f for f in os.listdir(input_dir) if f.lower().endswith(".npy")]
print(f"dB-scaled STFT → {len(files)} files | fs={fs}, nperseg={nperseg}, noverlap={noverlap}\n")

# 3) Compute & save features
for fname in tqdm(files, desc="Files", unit="file"):
    data = np.load(os.path.join(input_dir, fname))  # shape: (4, 5000)
    stft_ch = []
    for ch in range(4):
        _, _, Zxx = stft(
            data[ch], 
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            boundary=None
        )
        mag    = np.abs(Zxx)
        mag_db = 10.0 * np.log10(mag + eps)
        stft_ch.append(mag_db.astype(np.float32))

    feat = np.stack(stft_ch, axis=0)  # shape: (4, F, T)
    out_path = os.path.join(features_dir, fname)
    np.save(out_path, feat)

print(f"\nAll dB-scaled STFT features saved to: {features_dir}")
