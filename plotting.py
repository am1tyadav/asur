"""
plotting.py
Some utility functions to help plot data
author: https://github.com/am1tyadav
"""

import matplotlib.pyplot as plt
import librosa.display
import numpy as np


def dark_background() -> None:
    plt.style.use('dark_background')


def spectrogram(log_mel_spectrogram: np.ndarray, sample_rate: int) -> None:
    librosa.display.specshow(
        log_mel_spectrogram,
        sr=sample_rate,
        x_axis='time',
        y_axis='mel',
        cmap='inferno'
    )
