# src/features.py

import numpy as np
import torch

def stft_frame(x_frame: np.ndarray, n_fft: int = 2048) -> np.ndarray:
    """
    Compute a single-column STFT for a real-valued frame.
    Returns complex64 NumPy array of length F = n_fft // 2 + 1.
    """
    x_t = torch.from_numpy(x_frame.astype(np.float32))

    Xc = torch.stft(
        x_t,
        n_fft=n_fft,
        hop_length=n_fft,
        win_length=n_fft,
        window=torch.hann_window(n_fft),
        return_complex=True
    )

    return Xc[:, 0].cpu().numpy()  # shape: [F]
