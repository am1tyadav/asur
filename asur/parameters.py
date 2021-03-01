"""
parameters.py
author: https://github.com/am1tyadav
"""

from abc import ABC


class Parameters(ABC):
    """
    Parameters for extracting audio features
    duration: duration of the audio in seconds
    frame_size: number of samples in a frame
    hop_size: step size taken between two samples
    is_mono: if the audio is mono or not
    audio: Audio, numpy array
    """
    def __init__(self, **kwargs):
        self.duration = kwargs.get('duration')
        self.sample_rate = kwargs.get('sample_rate') or 16000
        self.frame_size = kwargs.get('frame_size') or 1024
        self.hop_size = kwargs.get('hop_size') or 512
        self.n_mels = kwargs.get('n_mels') or 256
        self.is_mono = kwargs.get('is_mono') or True
        self.audio = None
