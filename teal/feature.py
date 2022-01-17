import tensorflow as tf
from tensorflow.keras import layers
from teal.utils import tf_log10


class Normalize(layers.Layer):
    def __init__(self):
        super(Normalize, self).__init__()

    def call(self, inputs, *args, **kwargs):
        _abs = tf.abs(inputs)
        _max = tf.reduce_max(_abs, axis=1)
        _normalized = inputs / _max
        return _normalized


class STFT(layers.Layer):
    def __init__(self, n_fft: int, hop_length: int):
        super(STFT, self).__init__()

        self._n_fft = n_fft
        self._hop_length = hop_length

    def call(self, inputs, *args, **kwargs):
        return tf.signal.stft(
            inputs,
            frame_length=self._n_fft,
            frame_step=self._hop_length
        )

    def get_config(self):
        config = super(STFT, self).get_config()
        config.update({
            "_n_fft": self._n_fft,
            "_hop_length": self._hop_length,
        })
        return config


class LogMelSpectrogram(STFT):
    def __init__(
            self,
            sample_rate: int,
            n_fft: int,
            hop_length: int,
            n_mels: int,
            epsilon: float = 1e-9
    ):
        super(LogMelSpectrogram, self).__init__(n_fft=n_fft, hop_length=hop_length)

        self._sample_rate = sample_rate
        self._n_mels = n_mels
        self._epsilon = epsilon
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
        _stft = super(LogMelSpectrogram, self).call(inputs, *args, **kwargs)
        _spec = tf.abs(_stft)
        _mel_spec = tf.matmul(tf.square(_spec), self._lin_to_mel_matrix)
        return tf_log10(_mel_spec + self._epsilon)

    def get_config(self):
        config = super(LogMelSpectrogram, self).get_config()
        config.update({
            "_sample_rate": self._sample_rate,
            "_n_mels": self._n_mels,
            "_epsilon": self._epsilon
        })
        return config
