#!/usr/bin/env python3
import os
import random
import pickle

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter, OrderedDict
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
PROJECT            = r"C:\Users\csio\doa_project"
FEATURES_DIR       = os.path.join(PROJECT, "features")          # your 4×F×T .npy folder
LABELS_CSV         = os.path.join(PROJECT, "labels.csv")        # master labels.csv
OUTPUT_DIR         = os.path.join(PROJECT, "outputs_all_halfed") # new output folder, no clash
CHECKPOINT_PATH    = os.path.join(OUTPUT_DIR, "best_all.pt")
HISTORY_PATH       = os.path.join(OUTPUT_DIR, "history_all.pkl")
SPLIT_DIR          = os.path.join(PROJECT, "splits_all_halfed")

BATCH_SIZE = 8
LR         = 1e-3
EPOCHS     = 50
WD         = 1e-4
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Speed up on GPU
torch.backends.cudnn.benchmark = True

# ─── DATASET ────────────────────────────────────────────────────────────────────
class STFTDataset(Dataset):
    def __init__(self, features_dir, filenames, labels, noise_files=None):
        self.features_dir = features_dir
        self.filenames    = filenames
        self.labels       = labels
        self.noise_files  = noise_files

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fn    = self.filenames[idx]
        x_np  = np.load(os.path.join(self.features_dir, fn)).astype(np.float32)
        label = self.labels[idx]
        # optional: mix in a random noise example
        if self.noise_files and label != 0 and random.random() < 0.3:
            nf   = random.choice(self.noise_files)
            n_np = np.load(os.path.join(self.features_dir, nf)).astype(np.float32)
            snr  = 10**(random.uniform(-5,5)/20)
            x_np = x_np + n_np * snr
        return torch.from_numpy(x_np), torch.tensor(label, dtype=torch.long)

# ─── MODEL ──────────────────────────────────────────────────────────────────────
class CRNN(nn.Module):
    def __init__(self, in_ch=4, n_cls=37, dropout=0.5):
        super().__init__()
        # 3×Conv2d blocks
        layers, c = [], in_ch
        for out in (32, 64, 128):
            layers += [
                nn.Conv2d(c, out, 3, padding=1),
                nn.BatchNorm2d(out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            ]
            c = out
        self.cnn = nn.Sequential(*layers)

        # GRU deferred init
        self.gru    = None
        self.hid    = 128
        self.bidir  = True
        self.layers = 2
        self.dp     = dropout

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hid * (2 if self.bidir else 1), n_cls)
        )

    def forward(self, x):
        # x: (B,4,F,T)
        c = self.cnn(x)                # → (B,128,F',T')
        B, Cn, Fp, Tp = c.shape

        # Initialize GRU on first forward pass
        if self.gru is None:
            self.gru = nn.GRU(
                input_size=Cn * Fp,
                hidden_size=self.hid,
                num_layers=self.layers,
                batch_first=True,
                bidirectional=self.bidir,
                dropout=self.dp if self.layers > 1 else 0.0
            ).to(x.device)

        # Reshape for GRU: (B, T', Cn*F')
        seq, _ = self.gru(c.permute(0, 3, 1, 2).reshape(B, Tp, Cn * Fp))
        final  = seq[:, -1, :]         # last time-step
        return self.classifier(final)

# ─── HELPERS ────────────────────────────────────────────────────────────────────
def save_confusion_matrix(cm, labels, path):
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, cmap=plt.cm.Blues, interpolation='nearest')
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=90, fontsize=6)
    plt.yticks(ticks, labels, fontsize=6)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

# ─── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    # 0) Create output directories (so nothing from previous runs is overwritten)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SPLIT_DIR,   exist_ok=True)

    # 1) Load master labels.csv and filter to elevation ∈ {0,30,60} OR noise (elevation == -1)
    df  = pd.read_csv(LABELS_CSV)
    df0 = df[df.elevation.isin([0, 30, 60]) | (df.elevation == -1)].reset_index(drop=True)

    # 2) HALF-SAMPLING: randomly pick 50% of each (elevation, azimuth) group
    #    This is reproducible by setting random_state=42.
    df_sampled = (
        df0
        .groupby(["elevation", "azimuth"], sort=False, group_keys=False)
        .apply(lambda grp: grp.sample(frac=0.5, random_state=42))
        .reset_index(drop=True)
    )

    # 3) Extract arrays from this down-sampled DataFrame
    fns  = df_sampled.filename.values
    elev = df_sampled.elevation.values
    az   = df_sampled.azimuth.values

    # 4) Build an “ordered” list of unique (elevation, azimuth) pairs (including noise as (-1,-1))
    dist    = Counter(zip(elev, az))
    ordered = OrderedDict(
        sorted(
            dist.items(),
            key=lambda x: (x[0][0] < 0, x[0][0], x[0][1])
        )
    )
    unique_pairs = list(ordered.keys())   # e.g. [(-1,-1), (0,0), (0,30), …, (60,330)]

    # 5) Create mapping (elev,az) → integer class index, and index → label string
    cls_map   = {pair: i for i, pair in enumerate(unique_pairs)}
    idx2pair  = {i: pair for i, pair in enumerate(unique_pairs)}
    idx2label = {
        i: ("Noise" if a < 0 else f"{e}°/{a}°")
        for i, (e, a) in idx2pair.items()
    }

    # 6) Convert each row’s (elev, az) into an integer class label
    cls = np.array([cls_map[(e, a)] for e, a in zip(elev, az)], dtype=np.int64)

    # 7) Print class counts after the 50% down-sample
    print("\nClass counts (after 50% down‐sampling):")
    for (e, a), cnt in ordered.items():
        lbl = "Noise" if a < 0 else f"{e}°/{a}°"
        print(f"  {lbl:12s} {cnt}")

    # 8) Stratified 80/10/10 split on the down-sampled set
    s1 = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=42)
    tv_idx, test_idx = next(s1.split(fns, cls))

    s2 = StratifiedShuffleSplit(n_splits=1, test_size=0.1111, random_state=42)
    train_idx, val_idx = next(s2.split(fns[tv_idx], cls[tv_idx]))

    splits = {
        "train": len(train_idx),
        "val":   len(val_idx),
        "test":  len(test_idx)
    }
    print("\nSplits (80/10/10) on down‐sampled data:", splits)

    # 9) Save out CSVs of those splits (filename + azimuth + elevation)
    train_abs = tv_idx[train_idx]
    val_abs   = tv_idx[val_idx]
    test_abs  = test_idx

    df_sampled.iloc[train_abs][["filename", "azimuth", "elevation"]].to_csv(
        os.path.join(SPLIT_DIR, "train_all.csv"), index=False
    )
    df_sampled.iloc[val_abs][["filename", "azimuth", "elevation"]].to_csv(
        os.path.join(SPLIT_DIR, "val_all.csv"), index=False
    )
    df_sampled.iloc[test_abs][["filename", "azimuth", "elevation"]].to_csv(
        os.path.join(SPLIT_DIR, "test_all.csv"), index=False
    )

    # 10) Build “noise files” list (any row where azimuth < 0)
    noise_files = [
        fn for fn, (e_, a_) in zip(fns, zip(elev, az))
        if a_ < 0
    ]

    # 11) Create Dataset / DataLoader objects
    train_ds = STFTDataset(
        FEATURES_DIR,
        fns[tv_idx][train_idx],
        cls[tv_idx][train_idx],
        noise_files
    )
    val_ds   = STFTDataset(
        FEATURES_DIR,
        fns[tv_idx][val_idx],
        cls[tv_idx][val_idx]
    )
    test_ds  = STFTDataset(
        FEATURES_DIR,
        fns[test_idx],
        cls[test_idx]
    )

    # 12) WeightedRandomSampler on the training set to mitigate class imbalance
    train_class_counts = np.bincount(cls[tv_idx][train_idx])
    samp_w = 1.0 / train_class_counts.astype(np.float32)

    noise_class = cls_map.get((-1, -1), None)
    if noise_class is not None:
        # Clamp noise weight to ≤ 5× next highest
        nw = samp_w[noise_class]
        nxt = np.max(np.delete(samp_w, noise_class))
        samp_w[noise_class] = min(nw, nxt * 5.0)

    weights = samp_w[cls[tv_idx][train_idx]]
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(train_idx),
        replacement=True
    )

    train_ld = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=4 if DEVICE.type == "cuda" else 0,
        pin_memory=(DEVICE.type != "cpu")
    )
    val_ld = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4 if DEVICE.type == "cuda" else 0,
        pin_memory=(DEVICE.type != "cpu")
    )
    test_ld = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4 if DEVICE.type == "cuda" else 0,
        pin_memory=(DEVICE.type != "cpu")
    )

    # 13) Instantiate model / loss / optimizer / scheduler
    model     = CRNN(in_ch=4, n_cls=len(unique_pairs)).to(DEVICE)
    class_w   = torch.tensor(1.0 / train_class_counts, dtype=torch.float32, device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_w)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR,
        steps_per_epoch=len(train_ld),
        epochs=EPOCHS
    )

    history  = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val = 0.0

    # 14) TRAIN / VALIDATE loop
    for ep in range(1, EPOCHS + 1):
        # ——— TRAIN ———
        model.train()
        tloss = 0.0
        tcorrect = 0
        ttotal = 0

        for x, y in train_ld:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss   = criterion(logits, y)
            loss.backward()
            optimizer.step()
            scheduler.step()

            tloss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            tcorrect += (preds == y).sum().item()
            ttotal += x.size(0)

        tr_loss = tloss / ttotal
        tr_acc  = tcorrect / ttotal

        # ——— VALIDATE ———
        model.eval()
        vloss = 0.0
        vcorrect = 0
        vtotal = 0

        with torch.no_grad():
            for x, y in val_ld:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                l = criterion(logits, y)
                vloss += l.item() * x.size(0)
                vp = logits.argmax(dim=1)
                vcorrect += (vp == y).sum().item()
                vtotal += x.size(0)

        v_loss = vloss / vtotal
        v_acc  = vcorrect / vtotal

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(v_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(v_acc)

        print(
            f"Epoch {ep:02d}/{EPOCHS}  "
            f"Train: loss={tr_loss:.3f}, acc={tr_acc:.3f}  |  "
            f"Val:   loss={v_loss:.3f}, acc={v_acc:.3f}"
        )

        # Save best checkpoint if validation acc improved
        if v_acc > best_val:
            best_val = v_acc
            torch.save({"model_state_dict": model.state_dict()}, CHECKPOINT_PATH)

    # 15) Save history and plot curves
    pickle.dump(history, open(HISTORY_PATH, "wb"))

    # Loss plot
    plt.figure()
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.legend()
    plt.title("Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_all.png"), dpi=300)
    plt.close()

    # Accuracy plot
    plt.figure()
    plt.plot(history["train_acc"], label="Train Acc")
    plt.plot(history["val_acc"], label="Val Acc")
    plt.legend()
    plt.title("Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "acc_all.png"), dpi=300)
    plt.close()

    # 16) FINAL TEST PASS
    model.load_state_dict(torch.load(CHECKPOINT_PATH)["model_state_dict"])
    model.eval()

    all_preds = []
    all_true  = []

    with torch.no_grad():
        for x, y in test_ld:
            x = x.to(DEVICE)
            p = model(x).argmax(dim=1).cpu().numpy()
            all_preds.append(p)
            all_true.append(y.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_true)

    test_acc = accuracy_score(y_true, y_pred)
    print(f"\nTest Acc: {test_acc * 100:.2f}%")

    # 17) Confusion matrix with labels (font size small to avoid overlap)
    cm = confusion_matrix(y_true, y_pred)
    labels = [idx2label[i] for i in range(len(unique_pairs))]
    save_confusion_matrix(cm, labels, os.path.join(OUTPUT_DIR, "cm_all.png"))

    print(classification_report(y_true, y_pred, digits=4))

    # 18) Write per-file predictions to CSV
    df_out = pd.DataFrame({
        "filename":   test_ds.filenames,
        "true_class": [idx2label[i] for i in y_true],
        "pred_class": [idx2label[i] for i in y_pred],
        "correct":    (y_true == y_pred)
    })
    out_csv = os.path.join(OUTPUT_DIR, "test_predictions_all.csv")
    df_out.to_csv(out_csv, index=False)
    print("Wrote per-file results to:", out_csv)


if __name__ == "__main__":
    main()
