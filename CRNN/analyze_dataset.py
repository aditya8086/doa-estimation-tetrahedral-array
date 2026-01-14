import os
import numpy as np
from collections import Counter

# — PARAMETERS —
chunks_dir  = r"C:\Users\csio\doa_project\chunks"
dataset_dir = r"C:\Users\csio\doa_project\dataset"

def count_wav_chunks():
    wav_files = [
        f for f in os.listdir(chunks_dir)
        if f.lower().endswith(".wav")
    ]
    print(f"Total .wav chunks: {len(wav_files)}\n")

    # Count by prefix (e.g. dp010_base1, Noise, etc.)
    prefix_counts = Counter(f.split("_chunk")[0] for f in wav_files)
    print("Chunks per source file:")
    for prefix, cnt in prefix_counts.most_common():
        print(f"  {prefix:20s}: {cnt}")

def count_npy_stacks():
    npy_files = [
        f for f in os.listdir(dataset_dir)
        if f.lower().endswith(".npy")
    ]
    print(f"\nTotal .npy stacks: {len(npy_files)}\n")

    # Separate DOA vs Noise
    doa = [f for f in npy_files if f.startswith("dp")]
    noise = [f for f in npy_files if f.lower().startswith("noise")]
    print(f"DOA stacks   : {len(doa)}")
    print(f"Noise stacks : {len(noise)}")

    # Breakdown by elevation
    elev_counts = Counter(f[2: f.find("0") or 2] for f in doa)  
    # (this works since your dp files begin with e.g. dp0, dp30, dp60)
    print("\nDOA stacks by elevation:")
    for elev, cnt in elev_counts.items():
        print(f"  {elev:>2s}° : {cnt}")

def peek_one_stack():
    example = next(f for f in os.listdir(dataset_dir) if f.endswith(".npy"))
    arr = np.load(os.path.join(dataset_dir, example))
    print(f"\nExample stack: {example}")
    print(f"  shape = {arr.shape}, dtype = {arr.dtype}")

if __name__ == "__main__":
    count_wav_chunks()
    count_npy_stacks()
    peek_one_stack()

dataset_dir = r"C:\Users\csio\doa_project\dataset"
dp_files = [f for f in os.listdir(dataset_dir) if f.startswith("dp") and f.endswith(".npy")]

# Count per elevation prefix (0, 30, 60)
elev_counts = Counter(f[2:f.find("_")] for f in dp_files)

print(f"Total DP .npy stacks: {len(dp_files)}\n")
for elev, count in sorted(elev_counts.items(), key=lambda x: int(x[0])):
    print(f" Elevation {elev}° : {count} files")
