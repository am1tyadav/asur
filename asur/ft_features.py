"""
ft_features.py
author: https://github.com/am1tyadav
"""

import numpy as np
import librosa
import librosa.display

from asur.parameters import Parameters


class FTFeatures(Parameters):
    """
    Fourier transform based features
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fft = None

    def _load_fft(self) -> None:
        if not self.fft:
            self.fft = np.squeeze(np.fft.fft(self.audio))

    def magnitude(self, limit: float = 0.4) -> (np.ndarray, np.ndarray):
        assert 0 < limit <= 1., 'Limit must be between 0 and 1'
        fft_length = self.fft.shape[0]
        end_index = int(fft_length * limit)
        frequencies = np.linspace(0, self.sample_rate, fft_length)[:end_index]
        magnitudes = np.abs(self.fft)[:end_index]
        return frequencies, magnitudes

    def mel_spectrogram(self) -> np.ndarray:
        return np.squeeze(librosa.feature.melspectrogram(
            self.audio,
            sr=self.sample_rate,
            n_fft=self.frame_size,
            hop_length=self.hop_size,
            n_mels=self.n_mels
        ))

    def log_mel_spectrogram(self) -> np.ndarray:
        return librosa.power_to_db(self.mel_spectrogram())
