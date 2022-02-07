"""utils.py

Common utilities for teal
"""

import os
import tensorflow as tf
from tensorflow.keras.layers import Layer


def load_audio(file_path: str, expected_sr: int) -> tf.Tensor:
    """Load and return an audio wav file as a tf.Tensor

    Args:
        file_path: str - Absolute path of the wav file
        expected_sr: int - Expected sample rate of the wav file

    Returns:
        Loaded audio as a tf.Tensor
    """
    assert os.path.isfile(file_path), f"No file found at {file_path}"

    with open(file_path, "rb") as audio_file:
        audio_data = audio_file.read()

    audio, sample_rate = tf.audio.decode_wav(audio_data)
    audio = tf.squeeze(audio, axis=1)

    assert sample_rate == expected_sr
    return audio


def get_mel_filter_bank(sample_rate: int, n_fft: int, n_mels: int) -> tf.Tensor:
    return tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=n_mels,
        num_spectrogram_bins=n_fft // 2 + 1,
        sample_rate=sample_rate,
        lower_edge_hertz=0,
        upper_edge_hertz=sample_rate / 2,
    )
