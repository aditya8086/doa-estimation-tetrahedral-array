## CRNN-Based 3D DoA Estimation

This folder contains the CRNN pipeline described in the report for discrete
direction classification using multichannel STFT features.

### Pipeline
1. Stack 4-channel audio recordings
2. Chunk into fixed 0.25-second segments (4 × 4000)
3. Compute STFT (FFT=1024, 50% overlap)
4. Use dB-scaled spectrograms as input
5. Train CRNN with Conv2D + BiGRU layers

### Model Characteristics
- Input shape: (4, F, T)
- Output: 36 direction classes
- Loss: Cross-entropy
- Optimizer: Adam / AdamW
- Parameters: ~103K

### Notes
- Complex STFT variants are included for analysis
- Primary model uses magnitude (dB) STFT due to superior performance
