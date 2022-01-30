"""STFT

Computes short time fourier transform on input audio
"""

import tensorflow as tf
from tensorflow.keras import layers
from teal.utils import get_mel_filter_bank


class AudioToSTFT(layers.Layer):
    """STFT

    Computes short time fourier transform on input audio
    """

    def __init__(self, n_fft: int, hop_length: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._n_fft = n_fft
        self._hop_length = hop_length

    def call(self, inputs, *args, **kwargs):
        return tf.signal.stft(
            inputs, frame_length=self._n_fft, frame_step=self._hop_length
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "_n_fft": self._n_fft,
                "_hop_length": self._hop_length,
            }
        )
        return config


class AudioToSpectrogram(AudioToSTFT):
    """Spectrogram

    Computes power spectrum of input audio
    """

    def __init__(
        self, n_fft: int, hop_length: int, *args, power: float = 2.0, **kwargs
    ):
        super().__init__(n_fft=n_fft, hop_length=hop_length, *args, **kwargs)

        self._power = power

    def call(self, inputs, *args, **kwargs):
        _stft = super().call(inputs, *args, **kwargs)
        return tf.math.pow(tf.abs(_stft), self._power)

    def get_config(self):
        config = super().get_config()
        config.update({"_power": self._power})
        return config


class AudioToMelSpectrogram(AudioToSpectrogram):
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
        power: float = 2.0,
        **kwargs
    ):
        super().__init__(n_fft, hop_length, *args, power=power, **kwargs)

        self._sample_rate = sample_rate
        self._n_mels = n_mels
        self._lin_to_mel_matrix = None

    def build(self, input_shape):
        self._lin_to_mel_matrix = get_mel_filter_bank(
            self._sample_rate, self._n_fft, self._n_mels
        )

    def call(self, inputs, *args, **kwargs):
        _spec = super().call(inputs, *args, **kwargs)
        return tf.matmul(_spec, self._lin_to_mel_matrix)

    def get_config(self):
        config = super().get_config()
        config.update({"_sample_rate": self._sample_rate, "_n_mels": self._n_mels})
        return config
