"""STFT to Mag and Phase

Converts STFT to mag and phase
"""

import tensorflow as tf
from tensorflow.keras import layers


class STFTToSpecAndPhase(layers.Layer):
    def __init__(self, *args, power: float = 2.0, **kwargs):
        super().__init__(*args, **kwargs)

        self._power = power

    def call(self, inputs, *args, **kwargs):
        _mag = tf.math.pow(tf.abs(inputs), self._power)
        _phase = tf.math.imag(inputs)
        return [_mag, _phase]

    def get_config(self):
        config = super().get_config()
        config.update({"_power": self._power})
        return config


class STFTToSpectrogram(STFTToSpecAndPhase):
    """STFT to Spectrogram

    Converts STFT to mag
    """

    def __init__(self, *args, power: float = 2.0, **kwargs):
        super().__init__(*args, power=power, **kwargs)

    def call(self, inputs, *args, **kwargs):
        _mag, _ = super().call(inputs, *args, **kwargs)
        return _mag


class STFTToPhase(STFTToSpecAndPhase):
    """STFT to Phase

    Converts STFT to Phase
    """

    def __init__(self, *args, power: float = 2.0, **kwargs):
        super().__init__(*args, power=power, **kwargs)

    def call(self, inputs, *args, **kwargs):
        _, _phase = super().call(inputs, *args, **kwargs)
        return _phase


class STFTToAudio(layers.Layer):
    def __init__(self, n_fft: int, hop_length: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._n_fft = n_fft
        self._hop_length = hop_length

    def call(self, inputs, *args, **kwargs):
        return tf.signal.inverse_stft(
            inputs, frame_length=self._n_fft, frame_step=self._hop_length
        )

    def get_config(self):
        config = super().get_config()
        config.update({"_n_fft": self._n_fft, "_hop_length": self._hop_length})
