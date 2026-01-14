#!/usr/bin/env python3
import os
import soundfile as sf

segment_duration_sec = 0.25   
input_folder  = r"C:\Users\csio\Downloads\NEW DATA"
output_folder = r"C:\Users\csio\doa_project\chunks"

os.makedirs(output_folder, exist_ok=True)

wav_files = []
for root, _, files in os.walk(input_folder):
    for f in files:
        if f.lower().endswith(".wav"):
            wav_files.append(os.path.join(root, f))
print(f"→ Found {len(wav_files)} .wav files under {input_folder!r}")

if not wav_files:
    print("No .wav files found. Please double-check your path!")
    exit(1)

for path in sorted(wav_files):
    fname = os.path.basename(path)
    signal, sr = sf.read(path)
    segment_samples = int(segment_duration_sec * sr)
    total_segments = len(signal) // segment_samples

    for i in range(total_segments):
        start = i * segment_samples
        end   = start + segment_samples
        chunk = signal[start:end]

        base, _ = os.path.splitext(fname)
        chunk_fname = f"{base}_chunk{i+1:04d}.wav"
        chunk_path  = os.path.join(output_folder, chunk_fname)
        sf.write(chunk_path, chunk, sr)

    print(f"  {fname} → {total_segments} chunks")

print("\nDone! All chunks are in", output_folder)
