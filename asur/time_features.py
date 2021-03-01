"""
time_features.py
author: https://github.com/am1tyadav
"""

from typing import Callable

import numpy as np
import librosa

from asur.parameters import Parameters


class TimeFeatures(Parameters):
    """
    Time based audio features
    """
    def _env(self, func: Callable) -> np.ndarray:
        num_samples = self.audio.shape[0]
        return np.array([func(self.audio[i: i + self.frame_size])
                         for i in range(0, num_samples, self.hop_size)])

    def peak_envelope(self) -> np.ndarray:
        return self._env(max)

    def mean_envelope(self) -> np.ndarray:
        return self._env(np.mean)

    def rms_envelope(self) -> np.ndarray:
        def _root_mean_square(val):
            return np.sqrt(np.mean(np.square(val)))
        return self._env(_root_mean_square)

    def zero_crossing_rate(self) -> np.ndarray:
        return np.squeeze(librosa.feature.zero_crossing_rate(
            self.audio,
            frame_length=self.frame_size,
            hop_length=self.hop_size
        ))
