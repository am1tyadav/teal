"""STFT to Mag and Phase

Converts STFT to mag and phase
"""

import tensorflow as tf
from tensorflow.keras import layers


class STFTToSpectrogram(layers.Layer):
    """STFT to Spectrogram

    Converts STFT to mag
    """
    def __init__(self, *args, power: float = 2., **kwargs):
        super().__init__(*args, **kwargs)

        self._power = power

    def call(self, inputs, *args, **kwargs):
        return tf.math.pow(tf.abs(inputs), self._power)

    def get_config(self):
        config = super().get_config()
        config.update({
            "_power": self._power
        })
        return config


class STFTToPhase(layers.Layer):
    """STFT to Phase

    Converts STFT to Phase
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, inputs, *args, **kwargs):
        return tf.math.imag(inputs)
