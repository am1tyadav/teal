import os
import tensorflow as tf


def load_audio(file_path: str, expected_sr: int) -> tf.Tensor:
    assert os.path.isfile(file_path), f"No file found at {file_path}"

    with open(file_path, "rb") as f:
        audio_data = f.read()

    audio, sr = tf.audio.decode_wav(audio_data)
    audio = tf.squeeze(audio, axis=1)

    assert sr == expected_sr
    return audio
