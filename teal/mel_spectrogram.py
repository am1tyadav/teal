"""MelSpectrogram

Compute mel spectrogram of input audio
"""

import tensorflow as tf
from teal import Spectrogram


class MelSpectrogram(Spectrogram):
    """MelSpectrogram

    Compute mel spectrogram of input audio
    """

    def __init__(
            self,
            sample_rate: int,
            n_fft: int,
            hop_length: int,
            n_mels: int,
            *args,
            power: float = 2.,
            **kwargs
    ):
        super().__init__(
            n_fft,
            hop_length,
            *args,
            power=power,
            **kwargs
        )

        self._sample_rate = sample_rate
        self._n_mels = n_mels
        self._lin_to_mel_matrix = None

    def build(self, input_shape):
        self._lin_to_mel_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self._n_mels,
            num_spectrogram_bins=self._n_fft // 2 + 1,
            sample_rate=self._sample_rate,
            lower_edge_hertz=0,
            upper_edge_hertz=self._sample_rate / 2,
        )

    def call(self, inputs, *args, **kwargs):
        _spec = super().call(inputs, *args, **kwargs)
        return tf.matmul(_spec, self._lin_to_mel_matrix)

    def get_config(self):
        config = super().get_config()
        config.update({
            "_sample_rate": self._sample_rate,
            "_n_mels": self._n_mels
        })
        return config
