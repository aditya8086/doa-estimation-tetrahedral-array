import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# Corrected paths
CHUNK_DIR = r"C:\Users\csio\projects\new_gnn_project\4ChannelStackedChunk"
ANALYSIS_DIR = r"C:\Users\csio\projects\new_gnn_project\outputs_4ChannelStackedChunk"
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# Initialize a summary list
summary = []

# Discover all chunk files
chunk_files = sorted(f for f in os.listdir(CHUNK_DIR) if f.endswith(".npy"))

for fname in tqdm(chunk_files, desc="Analyzing stacked chunks"):
    path = os.path.join(CHUNK_DIR, fname)
    try:
        arr = np.load(path)
        shape = arr.shape
        stats = {
            "filename": fname,
            "shape": str(shape),
            "min": np.min(arr),
            "max": np.max(arr),
            "mean": np.mean(arr),
            "std": np.std(arr)
        }
    except Exception as e:
        stats = {
            "filename": fname,
            "shape": "ERROR",
            "min": "ERROR",
            "max": "ERROR",
            "mean": "ERROR",
            "std": "ERROR"
        }
    summary.append(stats)

# Save summary CSV
summary_df = pd.DataFrame(summary)
csv_path = os.path.join(ANALYSIS_DIR, "stacked_chunk_analysis.csv")
summary_df.to_csv(csv_path, index=False)

# Plot histogram of sample means for visualization
valid_means = summary_df[summary_df["mean"] != "ERROR"]["mean"].astype(float)
plt.hist(valid_means, bins=100)
plt.title("Histogram of Chunk Means")
plt.xlabel("Mean Amplitude")
plt.ylabel("Number of Files")
plt.tight_layout()
plot_path = os.path.join(ANALYSIS_DIR, "mean_histogram.png")
plt.savefig(plot_path, dpi=300)  
plt.close()
