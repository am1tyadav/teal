"""STFT

Computes short time fourier transform on input audio
"""

import tensorflow as tf
from tensorflow.keras import layers


class STFT(layers.Layer):
    """STFT

    Computes short time fourier transform on input audio
    """
    def __init__(self,
                 n_fft: int,
                 hop_length: int,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._n_fft = n_fft
        self._hop_length = hop_length

    def call(self, inputs, *args, **kwargs):
        return tf.signal.stft(
            inputs,
            frame_length=self._n_fft,
            frame_step=self._hop_length
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "_n_fft": self._n_fft,
            "_hop_length": self._hop_length,
        })
        return config
