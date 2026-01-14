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

PROJECT        = r"C:\Users\csio\doa_project"
FEATURES_DIR   = os.path.join(PROJECT, "features_complex")  
LABELS_CSV     = os.path.join(PROJECT, "labels.csv")
OUTPUT_DIR     = os.path.join(PROJECT, "outputs_complex")
CHECKPOINT     = os.path.join(PROJECT, "best_crnn_complex.pt")
HISTORY_PATH   = os.path.join(PROJECT, "history_complex.pkl")

BATCH_SIZE     = 128
LR             = 1e-3
EPOCHS         = 30
WD             = 1e-4
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.benchmark = True  # speedup

class STFTDataset(Dataset):
    def __init__(self, features_dir, filenames, labels, noise_files=None):
        self.features_dir = features_dir
        self.filenames    = filenames
        self.labels       = labels
        self.noise_files  = noise_files

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fn   = self.filenames[idx]
        x_np = np.load(os.path.join(self.features_dir, fn)).astype(np.float32)
        # x_np shape: (4, F, T, 2)

        # collapse real+imag into channels:
        # (4, 2, F, T) then reshape to (8, F, T)
        x_np = x_np.transpose(0, 3, 1, 2)       # (4,2,F,T)
        C, D, F, T = x_np.shape
        x_np = x_np.reshape(C*D, F, T)         # (8,F,T)

        label = self.labels[idx]

        if self.noise_files and label != 0 and random.random() < 0.3:
            nf   = random.choice(self.noise_files)
            n_np = np.load(os.path.join(self.features_dir, nf)).astype(np.float32)
            n_np = n_np.transpose(0,3,1,2).reshape(8, F, T)
            snr  = 10**(random.uniform(-5,5)/20)
            x_np = x_np + n_np * snr

        return torch.from_numpy(x_np), torch.tensor(label, dtype=torch.long)

class CRNN(nn.Module):
    def __init__(self, in_ch=8, n_cls=13, dropout=0.5):
        super().__init__()
        # CNN stack
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
        # x: (B,8,F,T)
        c = self.cnn(x)                   
        B, Cn, Fp, Tp = c.shape

        if self.gru is None:
            self.gru = nn.GRU(
                input_size=Cn*Fp,
                hidden_size=self.hid,
                num_layers=self.layers,
                batch_first=True,
                bidirectional=self.bidir,
                dropout=self.dp if self.layers>1 else 0.0
            ).to(x.device)

        seq, _ = self.gru(c.permute(0,3,1,2).reshape(B, Tp, Cn*Fp))
        final  = seq[:, -1, :]
        return self.classifier(final)

def save_confusion_matrix(cm, labels, path):
    plt.figure(figsize=(8,8))
    plt.imshow(cm, cmap=plt.cm.Blues, interpolation='nearest')
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45)
    plt.yticks(ticks, labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.colorbar(); plt.tight_layout()
    plt.savefig(path, dpi=300); plt.close()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df   = pd.read_csv(LABELS_CSV)
    df0  = df[(df.elevation==0)|(df.elevation==-1)].reset_index(drop=True)
    fns, az = df0.filename.values, df0.azimuth.values

    dist    = Counter(az)
    ordered = OrderedDict(sorted(dist.items(), key=lambda x:(x[0]<0, x[0])))
    unique  = list(ordered.keys())
    az2cls  = {a:i for i,a in enumerate(unique)}
    cls     = np.array([az2cls[a] for a in az])

    print("\nClass counts:")
    for a,c in ordered.items():
        lbl = "Noise" if a<0 else f"{a}°"
        print(f"  {lbl:<5s} {c}")

    s1 = StratifiedShuffleSplit(1, test_size=0.10, random_state=42)
    tv_idx, test_idx = next(s1.split(fns, cls))
    s2 = StratifiedShuffleSplit(1, test_size=0.1111, random_state=42)
    train_idx, val_idx = next(s2.split(fns[tv_idx], cls[tv_idx]))

    print("Splits:", {
        'train':len(train_idx),
        'val':  len(val_idx),
        'test': len(test_idx)
    })

    split_dir = os.path.join(PROJECT, "splits_complex")
    os.makedirs(split_dir, exist_ok=True)
    train_abs = tv_idx[train_idx]
    val_abs   = tv_idx[val_idx]
    test_abs  = test_idx

    df0.iloc[train_abs][["filename","azimuth","elevation"]]\
         .to_csv(os.path.join(split_dir,"train_split.csv"), index=False)
    df0.iloc[val_abs][["filename","azimuth","elevation"]]\
         .to_csv(os.path.join(split_dir,"val_split.csv"),   index=False)
    df0.iloc[test_abs][["filename","azimuth","elevation"]]\
         .to_csv(os.path.join(split_dir,"test_split.csv"),  index=False)
    print(f"→ saved splits to {split_dir}")

    noise_files = [fn for fn,l in zip(fns,az) if l==-1]
    train_ds = STFTDataset(FEATURES_DIR, fns[tv_idx][train_idx], cls[tv_idx][train_idx], noise_files)
    val_ds   = STFTDataset(FEATURES_DIR, fns[tv_idx][val_idx],   cls[tv_idx][val_idx])
    test_ds  = STFTDataset(FEATURES_DIR, fns[test_idx],          cls[test_idx])

    tcounts = np.bincount(cls[tv_idx][train_idx])
    samp_w  = 1.0 / tcounts
    noise_cls = az2cls[-1]
    samp_w[noise_cls] = min(samp_w[noise_cls], np.max(np.delete(samp_w,noise_cls))*5.0)
    weights = samp_w[cls[tv_idx][train_idx]]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_ld = DataLoader(train_ds, BATCH_SIZE, sampler=sampler,
                          num_workers=4 if DEVICE.type=="cuda" else 0,
                          pin_memory=(DEVICE.type!="cpu"))
    val_ld   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False,
                          num_workers=4 if DEVICE.type=="cuda" else 0,
                          pin_memory=(DEVICE.type!="cpu"))
    test_ld  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False,
                          num_workers=4 if DEVICE.type=="cuda" else 0,
                          pin_memory=(DEVICE.type!="cpu"))

    model     = CRNN(in_ch=8, n_cls=len(unique)).to(DEVICE)
    class_w   = torch.tensor(1.0/tcounts, dtype=torch.float32, device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_w)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR,
        steps_per_epoch=len(train_ld),
        epochs=EPOCHS
    )

    history, best_val = {'train_loss':[],'val_loss':[],'train_acc':[],'val_acc':[]}, 0.0

    for epoch in range(1, EPOCHS+1):
        model.train()
        running_loss=correct=total=0
        for x,y in train_ld:
            x,y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss   = criterion(logits,y)
            loss.backward(); optimizer.step(); scheduler.step()
            running_loss += loss.item()*x.size(0)
            preds        = logits.argmax(1)
            correct     += (preds==y).sum().item()
            total       += x.size(0)

        tr_l, tr_a = running_loss/total, correct/total

        model.eval(); vloss=vcorrect=vtot=0
        with torch.no_grad():
            for x,y in val_ld:
                x,y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                loss   = criterion(logits,y)
                vloss     += loss.item()*x.size(0)
                preds     = logits.argmax(1)
                vcorrect += (preds==y).sum().item()
                vtot     += x.size(0)
        v_l, v_a = vloss/vtot, vcorrect/vtot

        history['train_loss'].append(tr_l)
        history['val_loss'].append(v_l)
        history['train_acc'].append(tr_a)
        history['val_acc'].append(v_a)

        print(f"Epoch {epoch:02d}/{EPOCHS}  Train: loss={tr_l:.3f}, acc={tr_a:.3f}  |  Val: loss={v_l:.3f}, acc={v_a:.3f}")

        if v_a>best_val:
            best_val = v_a
            torch.save({'model_state_dict':model.state_dict()}, CHECKPOINT)

    pickle.dump(history, open(HISTORY_PATH,"wb"))
    plt.figure(); plt.plot(history['train_loss'],label='Train'); plt.plot(history['val_loss'],label='Val')
    plt.legend(); plt.title("Loss"); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,"loss.png"),dpi=300); plt.close()

    plt.figure(); plt.plot(history['train_acc'],label='Train'); plt.plot(history['val_acc'],label='Val')
    plt.legend(); plt.title("Accuracy"); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR,"acc.png"),dpi=300); plt.close()

    model.load_state_dict(torch.load(CHECKPOINT)['model_state_dict'])
    model.eval(); all_p, all_t = [],[]
    with torch.no_grad():
        for x,y in test_ld:
            x = x.to(DEVICE)
            preds = model(x).argmax(1).cpu().numpy()
            all_p.append(preds); all_t.append(y.cpu().numpy())

    y_pred = np.concatenate(all_p)
    y_true = np.concatenate(all_t)
    test_acc = accuracy_score(y_true,y_pred)
    print(f"\nTest Accuracy: {test_acc*100:.2f}%")
    cm = confusion_matrix(y_true,y_pred)
    print("Confusion Matrix:\n", cm)
    print(classification_report(y_true,y_pred,digits=4))

    labels_txt = ["Noise"] + [f"{a}°" for a in unique if a>=0]
    save_confusion_matrix(cm, labels_txt, os.path.join(OUTPUT_DIR,"confusion_matrix.png"))

    df_out = pd.DataFrame({
      "filename":     test_ds.filenames,
      "true_azimuth": [ unique[i] for i in y_true ],
      "pred_azimuth": [ unique[i] for i in y_pred ],
      "correct":      (y_true==y_pred)
    })
    out_csv = os.path.join(OUTPUT_DIR,"test_predictions_complex.csv")
    df_out.to_csv(out_csv,index=False)
    print("Wrote per-file results to:", out_csv)


if __name__=="__main__":
    main()
