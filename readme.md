# 3D Direction of Arrival Estimation Using a Tetrahedral Microphone Array

This repository contains the complete implementation of two deep learning models developed
during an internship at CSIR-CSIO, Chandigarh, for three-dimensional Direction of Arrival (DoA)
estimation using a custom-built 4-channel tetrahedral microphone array.

The work is based on real multichannel recordings collected in an anechoic environment and
presents a comparative study between a classification-based CRNN and a regression-based
GNN-inspired MLP model.

## Models Implemented

### 1. CRNN-Based DoA Estimation
- Input: 0.25s multichannel audio chunks
- Features: dB-scaled STFT spectrograms (4 × F × T)
- Task: Discrete direction classification (36 classes)
- Architecture: Conv2D + BatchNorm + BiGRU
- Strength: High accuracy and stable convergence

### 2. GNN-Inspired Pairwise MLP
- Input: Single-frame complex STFTs
- Features: 12 ordered microphone-pair feature vectors
- Task: Continuous azimuth and elevation regression
- Architecture: Pairwise MLP + Fusion MLP
- Strength: Lightweight, flexible, regression-capable

## Microphone Array Geometry
A tetrahedral microphone array with 8 cm spacing between all microphones was used,
enabling full 3D spatial localization (azimuth + elevation).

## Dataset
The dataset was recorded using a physical tetrahedral microphone array in an anechoic
chamber at CSIR-CSIO. It consists of recordings from 36 unique directions
(12 azimuth × 3 elevation).

Due to institutional and privacy constraints, the dataset is not publicly available.
All preprocessing, training, and evaluation code is provided for reproducibility.

## Internship Context
- Organization: CSIR–CSIO, Chandigarh
- Duration: April 2025 – June 2025
- Supervisor: Dr. Ripul Ghosh, Principal Scientist

## Note
This repository is intended for research, academic evaluation, and reproducibility.
