"""Ready to use examples of tf.keras models for audio preprocessing and data augmentation

Functions:
    create_log_mel_spectrogram_model
    create_data_augmentation_model
"""

import tensorflow as tf
from teal import augment, feature


def create_log_mel_spectrogram_model(
        num_samples: int = 66150,
        sample_rate: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128) -> tf.keras.models.Model:
    """Create and return a log mel spectrogram model

    Args:
        num_samples: int - Number of audio samples expected as input
        sample_rate: int - Sample rate of input audio
        n_fft: int - Number of fft bins
        hop_length: int - Hop length in number of samples
        n_mels: int - Number of mel filter banks to use

    Returns:
        A tf.keras model
    """
    _model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(num_samples, )),
        feature.NormalizeAudio(),
        feature.MelSpectrogram(sample_rate, n_fft, hop_length, n_mels),
        feature.PowerToDb(),
        feature.NormalizeSpectrum()
    ])
    return _model


def create_data_augmentation_model(num_samples: int = 66150) -> tf.keras.models.Model:
    """Create and return a tf.keras model for audio data augmentation

    This model applies InversePolarity, RandomGain and RandomNoise augmentations

    Args:
         num_samples: int - Number of samples expected in audio input

    Returns:
        A tf.keras model
    """
    _model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(num_samples, )),
        augment.InversePolarity(0.5),
        augment.RandomGain(0.5),
        augment.RandomNoise(0.5),
        augment.RandomGain(0.5)
    ])
    return _model
