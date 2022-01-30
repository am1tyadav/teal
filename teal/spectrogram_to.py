"""MelSpectrogram

Compute mel spectrogram of input audio
"""

import tensorflow as tf
from tensorflow.keras import layers
from teal.utils import get_mel_filter_bank


class SpectrogramToMelSpec(layers.Layer):
    """SpectrogramToMel

    Compute mel spectrogram of input power spectrum
    """

    def __init__(self, sample_rate: int, n_fft: int, n_mels: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._sample_rate = sample_rate
        self._n_fft = n_fft
        self._n_mels = n_mels
        self._lin_to_mel_matrix = None

    def build(self, input_shape):
        self._lin_to_mel_matrix = get_mel_filter_bank(
            self._sample_rate, self._n_fft, self._n_mels
        )

    def call(self, inputs, *args, **kwargs):
        return tf.matmul(inputs, self._lin_to_mel_matrix)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "_sample_rate": self._sample_rate,
                "_n_fft": self._n_fft,
                "_n_mels": self._n_mels,
            }
        )
        return config
