#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

PAIR_DIR    = r"C:\Users\csio\projects\new_gnn_project\pairs"
SPLIT_DIR   = r"C:\Users\csio\projects\new_gnn_project\splits"
OUTPUT_DIR  = r"C:\Users\csio\projects\new_gnn_project\outputs_normalized"
BATCH_SIZE  = 32
EPOCHS      = 100
LR          = 1e-3
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DOADataset(Dataset):
    def __init__(self, csv_path, pair_dir):
        self.df = pd.read_csv(csv_path)
        self.pair_dir = pair_dir

        existing_files = set(os.listdir(pair_dir))
        self.df = self.df[self.df["filename"].isin(existing_files)].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = np.load(os.path.join(self.pair_dir, row.filename)).astype(np.float32)
        y = np.array([
            row["azimuth"] / 360.0,
            row["elevation"] / 90.0
        ], dtype=np.float32)
        return torch.tensor(x), torch.tensor(y)

class PairwiseMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        B, P, D = x.shape
        x = x.view(B * P, D)
        out = self.mlp(x)
        return out.view(B, P, -1)

class FusionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x):
        return self.mlp(x.view(x.size(0), -1))
    
def train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_ds = DOADataset(os.path.join(SPLIT_DIR, "train.csv"), PAIR_DIR)
    val_ds   = DOADataset(os.path.join(SPLIT_DIR, "val.csv"), PAIR_DIR)
    test_ds  = DOADataset(os.path.join(SPLIT_DIR, "test.csv"), PAIR_DIR)

    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_ld   = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_ld  = DataLoader(test_ds, batch_size=BATCH_SIZE)

    x_sample, _ = train_ds[0]
    input_dim = x_sample.shape[-1]
    pair_mlp  = PairwiseMLP(input_dim=input_dim, hidden_dim=256, out_dim=64).to(DEVICE)
    fusion_mlp = FusionMLP(input_dim=64*12).to(DEVICE)

    from torchinfo import summary
    print(summary(pair_mlp, input_size=(BATCH_SIZE, 12, input_dim)))
    print(summary(fusion_mlp, input_size=(BATCH_SIZE, 12, 64)))

    optimizer = torch.optim.AdamW(
    list(pair_mlp.parameters()) + list(fusion_mlp.parameters()), lr=LR)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    train_losses, val_losses = [], []

    for epoch in range(EPOCHS):
        pair_mlp.train(); fusion_mlp.train()
        total_loss = 0
        for xb, yb in train_ld:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out = fusion_mlp(pair_mlp(xb))
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        train_losses.append(total_loss / len(train_ds))

        pair_mlp.eval(); fusion_mlp.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_ld:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out = fusion_mlp(pair_mlp(xb))
                val_loss += loss_fn(out, yb).item() * xb.size(0)
        val_losses.append(val_loss / len(val_ds))

        print(f"Epoch {epoch+1:02d}: Train Loss={train_losses[-1]:.4f}, Val Loss={val_losses[-1]:.4f}")
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            torch.save({
                "pairwise": pair_mlp.state_dict(),
                "fusion": fusion_mlp.state_dict()
            }, os.path.join(OUTPUT_DIR, "best_model.pt"))

    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.title("MSE Loss (Normalized)")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"))
    plt.close()

    pair_mlp.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model.pt"))["pairwise"])
    fusion_mlp.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model.pt"))["fusion"])
    pair_mlp.eval(); fusion_mlp.eval()

    y_preds, y_trues, used_filenames = [], [], []
    with torch.no_grad():
        for i, (xb, yb) in enumerate(test_ld):
            xb = xb.to(DEVICE)
            out = fusion_mlp(pair_mlp(xb)).cpu().numpy()
            y_preds.append(out)
            y_trues.append(yb.numpy())
            used_filenames.extend(test_ds.df.iloc[i*BATCH_SIZE : i*BATCH_SIZE+len(xb)]["filename"].tolist())

    y_preds = np.concatenate(y_preds)
    y_trues = np.concatenate(y_trues)

    y_preds[:, 0] *= 360.0
    y_preds[:, 1] *= 90.0
    y_trues[:, 0] *= 360.0
    y_trues[:, 1] *= 90.0

    acc_az = np.mean(np.abs(y_preds[:, 0] - y_trues[:, 0]) <= 15) * 100
    acc_el = np.mean(np.abs(y_preds[:, 1] - y_trues[:, 1]) <= 10) * 100

    print(f"\nTest MAE: Azimuth = {mean_absolute_error(y_trues[:,0], y_preds[:,0]):.2f}°, Elevation = {mean_absolute_error(y_trues[:,1], y_preds[:,1]):.2f}°")
    print(f"Test Accuracy: Azimuth ±15° = {acc_az:.2f}%, Elevation ±10° = {acc_el:.2f}%")

    az_mae = np.abs(y_preds[:, 0] - y_trues[:, 0])
    el_mae = np.abs(y_preds[:, 1] - y_trues[:, 1])

    print(f"Azimuth MAE: {az_mae.mean():.2f}° ± {az_mae.std():.2f}")
    print(f"Elevation MAE: {el_mae.mean():.2f}° ± {el_mae.std():.2f}")


    pred_df = pd.DataFrame({
        "filename": used_filenames,
        "azimuth_true": y_trues[:, 0],
        "elevation_true": y_trues[:, 1],
        "azimuth_pred": y_preds[:, 0],
        "elevation_pred": y_preds[:, 1],
    })
    pred_df.to_csv(os.path.join(OUTPUT_DIR, "test_predictions.csv"), index=False)
    print("Saved test predictions to CSV.")

    az_bins = list(range(0, 360, 30))   # 0° to 330°
    el_bins = [0, 30, 60]               # 3 elevations

    mae_grid = np.full((len(el_bins), len(az_bins)), np.nan)

    for i, el in enumerate(el_bins):
        for j, az in enumerate(az_bins):
            mask = (pred_df["azimuth_true"] == az) & (pred_df["elevation_true"] == el)
            if mask.any():
                az_err = np.abs(pred_df.loc[mask, "azimuth_true"] - pred_df.loc[mask, "azimuth_pred"])
                mae_grid[i, j] = az_err.mean()

    plt.figure(figsize=(10, 3))
    sns.heatmap(mae_grid, annot=True, fmt=".1f", cmap="YlOrRd", xticklabels=az_bins, yticklabels=el_bins, cbar_kws={'label': 'Azimuth MAE (°)'})
    plt.xlabel("Azimuth (°)")
    plt.ylabel("Elevation (°)")
    plt.title("Azimuth MAE Heatmap per Direction")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "azimuth_mae_heatmap.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    train()