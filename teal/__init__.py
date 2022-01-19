"""teal - TensorFlow Audio Layers

This module contains a number of TensorFlow layers specifically
written to be used with audio data

Base Layers:
    AugmentationLayer - Any data augmentation layer is subclassed from this

Feature layers:
    STFT
        Computes Short Time Fourier Transform
    Spectrogram
        Computes power spectrum
    MelSpectrogram
        Computes mel spectrogram
        Mel filter bank is computed once when the layer is built
    PowerToDb
        Scales the power spectrum to db
        Useful to create log mel spectrogram if used on MelSpectrogram output

Data augmentation layers
    InversePolarity
        Inverts polarity of input audio
    RandomGain
    RandomNoise

Other preprocessing layers
    Normalize
        NormalizeAudio
        NormalizeSpectrum
"""

# Base layers
from teal.augment import AugmentationLayer
# Feature layers
from teal.stft import STFT
from teal.spectrogram import Spectrogram
from teal.mel_spectrogram import MelSpectrogram
from teal.power_to_db import PowerToDb
# Augmentation layers
from teal.inverse_polarity import InversePolarity
from teal.random_gain import RandomGain
from teal.random_noise import RandomNoise
# Other preprocessing layers
from teal.normalize import Normalize, NormalizeAudio, NormalizeSpectrum
