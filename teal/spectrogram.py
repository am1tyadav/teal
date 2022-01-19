"""spectrogram

Computes power spectrum of input audio
"""


import tensorflow as tf
from teal.stft import STFT


class Spectrogram(STFT):
    """Spectrogram

    Computes power spectrum of input audio
    """
    def __init__(
            self,
            n_fft: int,
            hop_length: int,
            *args,
            power: float = 2.,
            **kwargs
    ):
        super().__init__(
            n_fft=n_fft,
            hop_length=hop_length,
            *args, **kwargs
        )

        self._power = power

    def call(self, inputs, *args, **kwargs):
        _stft = super().call(inputs, *args, **kwargs)
        return tf.math.pow(tf.abs(_stft), self._power)

    def get_config(self):
        config = super().get_config()
        config.update({
            "_power": self._power
        })
        return config
