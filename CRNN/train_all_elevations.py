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

from torchinfo import summary

PROJECT         = r"C:\Users\csio\doa_project"
FEATURES_DIR    = os.path.join(PROJECT, "features")       
LABELS_CSV      = os.path.join(PROJECT, "labels.csv")     
OUTPUT_DIR      = os.path.join(PROJECT, "outputs_all")    
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "best_all.pt")
HISTORY_PATH    = os.path.join(OUTPUT_DIR, "history_all.pkl")
SPLIT_DIR       = os.path.join(PROJECT, "splits_all")

BATCH_SIZE = 8
LR         = 1e-4              
EPOCHS     = 50
WD         = 1e-4
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.benchmark = True

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
        if self.noise_files and label == noise_label and random.random() < 0.3:
            nf   = random.choice(self.noise_files)
            n_np = np.load(os.path.join(self.features_dir, nf)).astype(np.float32)
            snr  = 10**(random.uniform(-5,5)/20)
            x_np = x_np + n_np * snr
        return torch.from_numpy(x_np), torch.tensor(label, dtype=torch.long)

class CRNN(nn.Module):
    def __init__(self, in_ch=4, n_cls=37, dropout=0.5):
        super().__init__()
        layers, c = [], in_ch
        for out in (32,64,128):
            layers += [
                nn.Conv2d(c, out, 3, padding=1),
                nn.BatchNorm2d(out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            ]
            c = out
        self.cnn = nn.Sequential(*layers)

        self.gru    = None
        self.hid    = 128
        self.bidir  = True
        self.layers = 2
        self.dp     = dropout

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hid * (2 if self.bidir else 1), n_cls)
        )

    def forward(self, x):
        c = self.cnn(x)                 
        B, Cn, Fp, Tp = c.shape
        if self.gru is None:
            self.gru = nn.GRU(
                Cn*Fp, self.hid, self.layers,
                batch_first=True,
                bidirectional=self.bidir,
                dropout=self.dp if self.layers>1 else 0.0
            ).to(x.device)
        seq, _ = self.gru(c.permute(0,3,1,2).reshape(B, Tp, Cn*Fp))
        return self.classifier(seq[:,-1,:])

def save_confusion_matrix(cm, labels, path):
    plt.figure(figsize=(12,10))
    plt.imshow(cm, cmap=plt.cm.Blues, interpolation='nearest')
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=90, fontsize=6)
    plt.yticks(ticks, labels, fontsize=6)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

noise_label = None

def main():
    global noise_label

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SPLIT_DIR,   exist_ok=True)

    df  = pd.read_csv(LABELS_CSV)
    df0 = df[df.elevation.isin([0,30,60]) | (df.elevation == -1)].reset_index(drop=True)
    fns, elev, az = df0.filename.values, df0.elevation.values, df0.azimuth.values

    dist    = Counter(zip(elev, az))
    ordered = OrderedDict(
        sorted(dist.items(), key=lambda x:(x[0][0]<0, x[0][0], x[0][1]))
    )
    unique_pairs = list(ordered.keys())            
    idx2pair     = {i: ea for i, ea in enumerate(unique_pairs)}
    idx2label    = {
        i: ("Noise" if ea == (-1,-1) else f"{ea[0]}°/{ea[1]}°")
        for i, ea in idx2pair.items()
    }
    cls_map = {ea: i for i, ea in idx2pair.items()}
    cls     = np.array([cls_map[(e,a)] for e,a in zip(elev, az)], dtype=np.int64)

    noise_label = cls_map[(-1, -1)]

    print("\nClass counts (all elevations + noise):")
    for (e,a), cnt in ordered.items():
        lbl = "Noise" if (e,a) == (-1,-1) else f"{e}°/{a}°"
        print(f"  {lbl:12s} {cnt}")

    s1 = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    temp_idx, test_idx = next(s1.split(fns, cls))
    s2 = StratifiedShuffleSplit(n_splits=1, test_size=0.1765, random_state=42)
    train_sub, val_sub = next(s2.split(fns[temp_idx], cls[temp_idx]))

    train_idx = temp_idx[train_sub]
    val_idx   = temp_idx[val_sub]


    print("\nSplit sizes:")
    print(f"  Train: {len(train_idx)}")
    print(f"  Val:   {len(val_idx)}")
    print(f"  Test:  {len(test_idx)}")

    train_df = pd.DataFrame({
        "filename":  df0.filename.values[train_idx],
        "azimuth":   df0.azimuth.values[train_idx],
        "elevation": df0.elevation.values[train_idx]
    })
    train_df.to_csv(os.path.join(SPLIT_DIR, "train_all.csv"), index=False)

    val_df = pd.DataFrame({
        "filename":  df0.filename.values[val_idx],
        "azimuth":   df0.azimuth.values[val_idx],
        "elevation": df0.elevation.values[val_idx]
    })
    val_df.to_csv(os.path.join(SPLIT_DIR, "val_all.csv"), index=False)

    test_df = pd.DataFrame({
        "filename":  df0.filename.values[test_idx],
        "azimuth":   df0.azimuth.values[test_idx],
        "elevation": df0.elevation.values[test_idx]
    })
    test_df.to_csv(os.path.join(SPLIT_DIR, "test_all.csv"), index=False)

    noise_files = [
        fn for fn, (e,a) in zip(fns, zip(elev, az))
        if (e,a) == (-1,-1)
    ]
    train_ds = STFTDataset(
        FEATURES_DIR,
        df0.filename.values[train_idx],
        cls[train_idx],
        noise_files
    )
    val_ds = STFTDataset(
        FEATURES_DIR,
        df0.filename.values[val_idx],
        cls[val_idx]
    )
    test_ds = STFTDataset(
        FEATURES_DIR,
        df0.filename.values[test_idx],
        cls[test_idx]
    )

    train_labels = cls[train_idx]
    tcounts = np.bincount(train_labels, minlength=len(unique_pairs))
    samp_w  = 1.0 / tcounts.astype(np.float32)

    nw = samp_w[noise_label]
    other_max = np.max(np.delete(samp_w, noise_label))
    samp_w[noise_label] = min(nw, other_max * 5.0)

    sampler = WeightedRandomSampler(
        weights=samp_w[train_labels],
        num_samples=len(train_labels),
        replacement=True
    )

    train_ld = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=4 if DEVICE.type == "cuda" else 0,
        pin_memory=(DEVICE.type != "cpu")
    )
    val_ld = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4 if DEVICE.type == "cuda" else 0,
        pin_memory=(DEVICE.type != "cpu")
    )
    test_ld = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4 if DEVICE.type == "cuda" else 0,
        pin_memory=(DEVICE.type != "cpu")
    )
    model = CRNN(in_ch=4, n_cls=len(unique_pairs)).to(DEVICE)
    summary(model, input_size=(BATCH_SIZE, 4, 513, 9))  # Added summary
    class_w   = torch.tensor(1.0 / tcounts, dtype=torch.float32, device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_w)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WD,
        betas=(0.9, 0.999)
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5
    )

    history, best_val = {
        'train_loss': [], 'val_loss': [],
        'train_acc':  [], 'val_acc': []
    }, 0.0

    for ep in range(1, EPOCHS + 1):
        # — train —
        model.train()
        total_loss = correct = total = 0
        for x_batch, y_batch in train_ld:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x_batch.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total   += x_batch.size(0)

        tr_l = total_loss / total
        tr_a = correct / total

        model.eval()
        vloss = vcorrect = vtot = 0
        with torch.no_grad():
            for x_batch, y_batch in val_ld:
                x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
                logits = model(x_batch)
                loss   = criterion(logits, y_batch)
                vloss    += loss.item() * x_batch.size(0)
                preds    = logits.argmax(dim=1)
                vcorrect += (preds == y_batch).sum().item()
                vtot     += x_batch.size(0)

        v_l = vloss / vtot
        v_a = vcorrect / vtot

        history['train_loss'].append(tr_l)
        history['val_loss'].append(v_l)
        history['train_acc'].append(tr_a)
        history['val_acc'].append(v_a)

        print(
            f"Epoch {ep:02d}/{EPOCHS}  "
            f"Train: loss={tr_l:.3f}, acc={tr_a:.3f}  |  "
            f"Val:   loss={v_l:.3f}, acc={v_a:.3f}"
        )
        scheduler.step(v_l)

        if v_a > best_val:
            best_val = v_a
            torch.save({'model_state_dict': model.state_dict()}, CHECKPOINT_PATH)

    pickle.dump(history, open(HISTORY_PATH, "wb"))
    plt.figure()
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'],   label='Val Loss')
    plt.legend(); plt.title("Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_all.png"), dpi=300)
    plt.close()

    plt.figure()
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'],   label='Val Acc')
    plt.legend(); plt.title("Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "acc_all.png"), dpi=300)
    plt.close()

    model.load_state_dict(torch.load(CHECKPOINT_PATH)['model_state_dict'])
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for x_batch, y_batch in test_ld:
            x_batch = x_batch.to(DEVICE)
            preds = model(x_batch).argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_trues.append(y_batch.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_trues)

    print(f"\nTest Acc: {accuracy_score(y_true, y_pred)*100:.2f}%")
    cm = confusion_matrix(y_true, y_pred)
    labels = [idx2label[i] for i in range(len(unique_pairs))]
    save_confusion_matrix(cm, labels, os.path.join(OUTPUT_DIR, "cm_all.png"))
    print(classification_report(y_true, y_pred, digits=4))

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
