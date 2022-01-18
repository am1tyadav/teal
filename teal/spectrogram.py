import tensorflow as tf
from teal.stft import STFT


class Spectrogram(STFT):
    def __init__(
            self,
            n_fft: int,
            hop_length: int,
            power: float = 2.
    ):
        super(Spectrogram, self).__init__(n_fft=n_fft, hop_length=hop_length)

        self._power = power

    def call(self, inputs, *args, **kwargs):
        _stft = super(Spectrogram, self).call(inputs, *args, **kwargs)
        return tf.math.pow(tf.abs(_stft), self._power)

    def get_config(self):
        config = super(Spectrogram, self).get_config()
        config.update({
            "_power": self._power
        })
        return config
