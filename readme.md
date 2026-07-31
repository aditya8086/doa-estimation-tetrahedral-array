# 3D Direction of Arrival Estimation Using a Tetrahedral Microphone Array

**Two deep-learning approaches to 3D sound source localization — a classification CRNN and a regression GNN-inspired MLP — benchmarked head-to-head on real anechoic-chamber recordings from a custom 4-channel tetrahedral microphone array.**

`PyTorch` · `CRNN` · `GNN-inspired MLP` · `STFT` · `Direction-of-Arrival` · `Spatial Audio`

This repository contains the complete implementation of two deep learning models developed during an internship at **CSIR-CSIO, Chandigarh**, for three-dimensional Direction of Arrival (DoA) estimation. The work is based on real multichannel recordings collected in an anechoic environment and presents a comparative study between a classification-based CRNN and a regression-based GNN-inspired MLP.

---

## Results

| Model | Task | Metric | Result |
|---|---|---|---|
| **CRNN** | Discrete direction classification (36 classes) | Azimuth accuracy | **~97.8%** |
| **GNN-inspired MLP** | Continuous azimuth–elevation regression | Azimuth MAE | **~2.3°** |
| | | Elevation MAE | **~0.9°** |
| | | Threshold accuracy | **>97.7%** |

The CRNN classifier proved highly accurate and stable in convergence across both Adam and AdamW optimizers and 16 / 20 kHz sampling rates. The GNN-inspired regressor is lightweight and parallelizable, and predicts continuous angles rather than discrete bins.

---

## Related publication

The GNN-inspired model developed here served as the **base architecture** for a subsequent publication:

> **Masked-Aware Directional Attention Network for DOA Estimation Under Sensor Failure Conditions** — *IEEE Sensors Letters*, presented at IEEE ASPCON 2026.

The published work (led by a PhD scholar at CSIR-CSIO) extended this base GNN model with a masked-attention mechanism to study robustness under microphone failure. This repository contains the original comparative CRNN / GNN study that preceded and underpinned that work.

---

## Models Implemented

### 1. CRNN-Based DoA Estimation
- **Input:** 0.25 s multichannel audio chunks
- **Features:** dB-scaled STFT spectrograms (4 × F × T)
- **Task:** Discrete direction classification (36 classes)
- **Architecture:** Conv2D + BatchNorm + BiGRU (~103K parameters)
- **Strength:** High accuracy and stable convergence

### 2. GNN-Inspired Pairwise MLP
- **Input:** Single-frame complex STFTs
- **Features:** 12 ordered microphone-pair feature vectors
- **Task:** Continuous azimuth and elevation regression
- **Architecture:** Pairwise MLP + Fusion MLP
- **Strength:** Lightweight, flexible, regression-capable

---

## Microphone Array Geometry

A tetrahedral microphone array with **8 cm spacing** between all microphones was used, enabling full 3D spatial localization (azimuth + elevation). The tetrahedral geometry provides a non-coplanar sensor arrangement, which is what makes elevation estimation tractable alongside azimuth.

---

## Dataset

The dataset was recorded using a physical tetrahedral microphone array in an anechoic chamber at CSIR-CSIO. It consists of recordings from **36 unique directions** (12 azimuth × 3 elevation).

Due to institutional and privacy constraints, the dataset itself is not publicly available. All preprocessing, training, and evaluation code is provided for reproducibility.

---

## Methodology notes

- **Feature representation.** STFT-based time-frequency representations computed per channel. The CRNN consumes dB-scaled magnitude spectrograms stacked across the 4 channels; the GNN-inspired model consumes complex STFT features arranged as 12 ordered microphone-pair vectors (all pairs of the 4-mic array).
- **Classification vs. regression.** The two models frame DoA differently on purpose: the CRNN discretizes the sphere into 36 labelled directions and classifies; the GNN-inspired MLP predicts continuous azimuth/elevation angles, trading a small amount of accuracy for angular resolution beyond the 36-bin grid.
- **Evaluation.** Both models are evaluated on identical data splits. The comparative study includes confusion-matrix analysis for the classifier and per-axis MAE for the regressor.

---

## Repository structure

```
doa-estimation-tetrahedral-array/
├── CRNN/                 # CRNN classification pipeline
│   ├── chunk_wavs.py, 0.25_wavs_to_(4,N)npy.py   # chunking to 0.25s segments
│   ├── extract_stft.py, extract_stft_complex.py  # STFT feature extraction
│   ├── features_labels.py, features_to_zip.py    # feature/label packaging
│   ├── analyze_*.py, compare_spectrograms.py     # analysis utilities
│   ├── train.py, train_all_elevations.py, train_complex.py
│   ├── visualize_stft.py
│   └── readme.md         # CRNN pipeline details
│
├── GNN/                  # GNN-inspired pairwise MLP pipeline
│   ├── chunk_audio.py, extract_stft.py           # per-frame STFT
│   ├── make_pairs.py, zip_pairs.py               # mic-pair feature construction
│   ├── features.py, analyze_pairs.py, analyze_features.py
│   ├── split_csv.py                              # train/val/test split
│   ├── train.py
│   ├── requirements.txt
│   └── readme.md         # GNN pipeline details
│
├── data/
│   └── readme.md         # dataset notes (data itself not public)
├── .gitignore
└── readme.md            # this file
```

Each model folder has its own README with pipeline-level detail:
- [`CRNN/readme.md`](CRNN/readme.md) — chunking, STFT features, Conv2D + BiGRU classifier
- [`GNN/readme.md`](GNN/readme.md) — per-frame complex STFT, 12 mic-pair features, pairwise + fusion MLP

---

## Internship Context

- **Organization:** CSIR–CSIO, Chandigarh
- **Duration:** April 2025 – June 2025
- **Supervisor:** Dr. Ripul Ghosh, Principal Scientist

---

## Note

This repository is intended for research, academic evaluation, and reproducibility.
