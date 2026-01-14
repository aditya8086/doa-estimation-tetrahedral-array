## GNN-Inspired Pairwise MLP for 3D DoA Estimation

This folder contains a graph-inspired regression model that learns spatial
relationships using pairwise microphone features.

### Pipeline
0. Frame-based chunking (4 × 2048)
1. Compute complex STFT per frame
2. Construct 12 ordered microphone-pair features
3. Split dataset into train/val/test CSVs
4. Train pairwise MLP + fusion MLP

### Feature Structure
Each microphone pair contains:
- Real & imaginary STFT components of both microphones
- 3D coordinates of both microphones

Final input shape: (12, 4F + 6)

### Model Characteristics
- Output: Continuous azimuth & elevation
- Loss: Mean Squared Error (MSE)
- Optimizer: AdamW
- Parameters: ~1.17M
