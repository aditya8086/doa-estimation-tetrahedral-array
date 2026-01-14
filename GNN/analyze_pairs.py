# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from tqdm import tqdm

# # Define your pairwise feature directory and analysis output folder
# PAIR_DIR = r"C:\Users\csio\projects\new_gnn_project\pairs"
# ANALYSIS_DIR = r"C:\Users\csio\projects\new_gnn_project\outputs_pairs"
# os.makedirs(ANALYSIS_DIR, exist_ok=True)

# # Initialize summary list
# summary = []

# # List all .npy files
# pair_files = sorted(f for f in os.listdir(PAIR_DIR) if f.endswith(".npy"))

# for fname in tqdm(pair_files, desc="Analyzing pairwise features"):
#     path = os.path.join(PAIR_DIR, fname)
#     try:
#         arr = np.load(path)
#         stats = {
#             "filename": fname,
#             "shape": str(arr.shape),
#             "min": np.min(arr),
#             "max": np.max(arr),
#             "mean": np.mean(arr),
#             "std": np.std(arr)
#         }
#     except Exception as e:
#         stats = {
#             "filename": fname,
#             "shape": "ERROR",
#             "min": "ERROR",
#             "max": "ERROR",
#             "mean": "ERROR",
#             "std": "ERROR"
#         }
#     summary.append(stats)

# # Create and save summary DataFrame
# summary_df = pd.DataFrame(summary)
# csv_path = os.path.join(ANALYSIS_DIR, "pair_feature_analysis.csv")
# summary_df.to_csv(csv_path, index=False)

# # Plot histogram of valid mean values
# valid_means = summary_df[summary_df["mean"] != "ERROR"]["mean"].astype(float)
# plt.hist(valid_means, bins=100)
# plt.title("Histogram of Pairwise Feature Means")
# plt.xlabel("Mean Amplitude")
# plt.ylabel("Number of Files")
# plt.tight_layout()
# plot_path = os.path.join(ANALYSIS_DIR, "pair_mean_histogram.png")
# plt.savefig(plot_path, dpi=300)
# plt.close()

import numpy as np
import matplotlib.pyplot as plt

x = np.load('/kaggle/input/2f-6pairs/pairs/dp01_...npy')  # replace with real file
plt.imshow(x, aspect='auto')
plt.title("Feature Visual Check")
plt.colorbar()
plt.show()
