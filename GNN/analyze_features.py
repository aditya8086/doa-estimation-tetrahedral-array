import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

FEATURE_DIR = r"C:\Users\csio\projects\new_gnn_project\features"
ANALYSIS_DIR = r"C:\Users\csio\projects\new_gnn_project\outputs_features"
os.makedirs(ANALYSIS_DIR, exist_ok=True)

summary = []
feature_files = sorted(f for f in os.listdir(FEATURE_DIR) if f.endswith(".npy"))

for fname in tqdm(feature_files, desc="Analyzing STFT features"):
    path = os.path.join(FEATURE_DIR, fname)
    try:
        arr = np.load(path)
        assert arr.ndim == 3 and arr.shape[0] == 4 and arr.shape[1] == 2
        stats = {
            "filename": fname,
            "shape": str(arr.shape),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr))
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

summary_df = pd.DataFrame(summary)
csv_path = os.path.join(ANALYSIS_DIR, "stft_feature_analysis.csv")
summary_df.to_csv(csv_path, index=False)

valid_means = summary_df[summary_df["mean"] != "ERROR"]["mean"].astype(float)
plt.hist(valid_means, bins=100)
plt.title("Histogram of STFT Feature Means")
plt.xlabel("Mean Amplitude")
plt.ylabel("Number of Files")
plt.tight_layout()
plot_path = os.path.join(ANALYSIS_DIR, "mean_histogram_stft.png")
plt.savefig(plot_path, dpi=300)
plt.close()
