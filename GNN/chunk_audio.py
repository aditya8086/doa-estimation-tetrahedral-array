import os
import numpy as np
from scipy.io import wavfile

# ─── CONFIG ───────────────────────────────────────────────────────────────
INPUT_DIR = r"C:\Users\csio\projects\new_gnn_project\NEW DATA"
OUTPUT_DIR = r"C:\Users\csio\projects\new_gnn_project\4ChannelStackedChunk"
N_FFT = 2048
HOP = 1024 

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── DISCOVER FILES ────────────────────────────────────────────────────────
all_files = []
for root, _, filenames in os.walk(INPUT_DIR):
    for f in filenames:
        if f.endswith(".wav") and f.startswith(("dp0", "dp30", "dp60", "Noise")):
            all_files.append(os.path.join(root, f))

# ─── GROUP PREFIXES ────────────────────────────────────────────────────────
dp_prefixes = sorted({
    os.path.basename(f).split("_", 1)[0][2:] for f in all_files
    if os.path.basename(f).startswith(("dp0", "dp30", "dp60"))
})

noise_groups = {}
for f in all_files:
    base = os.path.basename(f)
    if base.startswith("Noise") and base.endswith(".wav"):
        parts = base.split("_")
        if len(parts) == 2 and parts[1].startswith("base") or parts[1].startswith("top"):
            mic = parts[1].split(".")[0]
            noise_groups[mic] = f

# ─── CHUNK DP FILES ────────────────────────────────────────────────────────
for p in dp_prefixes:
    print(f"\nProcessing dp{p}...")

    names = [f"dp{p}_base1.wav", f"dp{p}_base2.wav",
             f"dp{p}_base3.wav", f"dp{p}_top.wav"]
    sigs = []
    sr0 = None
    for nm in names:
        full_path = next((f for f in all_files if f.endswith(nm)), None)
        if not full_path:
            raise FileNotFoundError(f"Missing mic file: {nm}")
        sr, data = wavfile.read(full_path)
        if sr0 is None:
            sr0 = sr
        elif sr != sr0:
            raise ValueError("Sample rate mismatch")
        if data.dtype != np.float32:
            data = data.astype(np.float32) / np.iinfo(data.dtype).max
        sigs.append(data)

    L = min(len(x) for x in sigs)
    num_frames = 1 + (L - N_FFT) // HOP

    for k in range(num_frames):
        start = k * HOP
        chunk = [s[start:start+N_FFT] for s in sigs]
        chunk = np.stack(chunk, axis=0)
        out_name = f"dp{p}_chunk{str(k+1).zfill(5)}.npy"
        np.save(os.path.join(OUTPUT_DIR, out_name), chunk)

    print(f"Saved {num_frames} chunks for dp{p}")

# ─── CHUNK NOISE FILES ─────────────────────────────────────────────────────
if len(noise_groups) == 4:
    print(f"\nProcessing Noise...")
    sigs = []
    sr0 = None
    for mic in ["base1", "base2", "base3", "top"]:
        full_path = noise_groups.get(mic, None)
        if not full_path:
            raise FileNotFoundError(f"Missing Noise_{mic}.wav")
        sr, data = wavfile.read(full_path)
        if sr0 is None:
            sr0 = sr
        elif sr != sr0:
            raise ValueError("Sample rate mismatch")
        if data.dtype != np.float32:
            data = data.astype(np.float32) / np.iinfo(data.dtype).max
        sigs.append(data)

    L = min(len(x) for x in sigs)
    num_frames = 1 + (L - N_FFT) // HOP

    for k in range(num_frames):
        start = k * HOP
        chunk = [s[start:start+N_FFT] for s in sigs]
        chunk = np.stack(chunk, axis=0)
        out_name = f"Noise_chunk{str(k+1).zfill(5)}.npy"
        np.save(os.path.join(OUTPUT_DIR, out_name), chunk)

    print(f"Saved {num_frames} noise chunks.")
else:
    print("Skipping Noise: not all 4 mic files found.")

print("\nDone chunking all recordings!")
