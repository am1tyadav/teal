"""utils.py

Common utilities for teal
"""

import os
import tensorflow as tf


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
