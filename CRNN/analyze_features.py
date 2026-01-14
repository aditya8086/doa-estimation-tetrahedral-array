import os
import numpy as np
import matplotlib.pyplot as plt

# — Adjust paths —
feature_dir = r"C:\Users\csio\doa_project\features"
output_dir  = r"C:\Users\csio\doa_project\plots_features"
os.makedirs(output_dir, exist_ok=True)

# Pick a sample file (change index as needed)
files = sorted(f for f in os.listdir(feature_dir) if f.endswith(".npy"))
sample = files[12000]  # change this index or use random.choice
tensor = np.load(os.path.join(feature_dir, sample))  # shape: (4, F, T)

# Plot (1 row, 4 columns — one per mic)
fig, axes = plt.subplots(1, 4, figsize=(20, 4))
for mic in range(4):
    axes[mic].imshow(tensor[mic], aspect='auto', origin='lower', cmap='viridis')
    axes[mic].set_title(f"Mic {mic+1}")
    axes[mic].set_ylabel("Freq Bins")
    axes[mic].set_xlabel("Time Frames")

plt.suptitle(f"dB-scaled STFT Features — {sample}", fontsize=16)
plt.tight_layout()

# Save the figure
out_path = os.path.join(output_dir, sample.replace(".npy", ".png"))
plt.savefig(out_path, dpi=300)
print(f"Saved plot to: {out_path}")
