"""
audio_file.py
author: https://github.com/am1tyadav
"""

import os
import librosa

from asur.time_features import TimeFeatures
from asur.ft_features import FTFeatures


class AudioFile(TimeFeatures, FTFeatures):
    """
    Defines the AudioFile class
    An AudioFile instance can encapsulate time based,
    frequency based and fourier transform based features
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.file_path = kwargs.get('file_path')
        assert self.file_path, 'File path can not be None'
        assert os.path.isfile(self.file_path), f'File path not found {self.file_path}'
        self._load_audio()
        self._load_fft()

    def _load_audio(self) -> None:
        if not self.audio:
            self.audio, _ = librosa.load(
                self.file_path,
                sr=self.sample_rate,
                duration=self.duration,
                mono=self.is_mono
            )
