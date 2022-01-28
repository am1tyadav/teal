"""teal - TensorFlow Audio Layers

This module contains a number of TensorFlow layers specifically
written to be used with audio data
"""
# Augmentation
from teal.augment import (
    AugmentationLayer, InversePolarity, NoiseBank,
    RandomGain, RandomNoise, PitchShift
)
# Transformation
from teal.audio_to import (
    AudioToSTFT, AudioToSpectrogram,
    AudioToMelSpectrogram
)
from teal.stft_to import (
    STFTToSpecAndPhase, STFTToSpectrogram, STFTToPhase
)
from teal.power_to_db import PowerToDb
from teal.db_to_power import DbToPower
from teal.spectrogram_to import SpectrogramToMelSpec
# Generic
from teal.normalize import NormalizeAudio, NormalizeSpectrum
