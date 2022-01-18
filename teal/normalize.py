from typing import Any
import tensorflow as tf
from tensorflow.keras import layers


class Normalize(layers.Layer):
    def __init__(self, axes: Any = 1, epsilon: float = 1e-10):
        super(Normalize, self).__init__()

        self._axes = axes
        self._epsilon = epsilon

    def call(self, inputs, *args, **kwargs):
        _max = tf.reduce_max(tf.abs(inputs), axis=self._axes)
        _max = tf.expand_dims(_max + self._epsilon, axis=-1)
        return inputs / _max

    def get_config(self):
        config = super(Normalize, self).get_config()
        config.update({
            "_axes": self._axes,
            "_epsilon": self._epsilon
        })


class NormalizeAudio(Normalize):
    def __init__(self):
        super(NormalizeAudio, self).__init__(axes=1)

    def call(self, inputs, *args, **kwargs):
        return super(NormalizeAudio, self).call(inputs, *args, **kwargs)


class NormalizeSpectrum(Normalize):
    def __init__(self):
        super(NormalizeSpectrum, self).__init__(axes=(1, 2))

    def call(self, inputs, *args, **kwargs):
        return super(NormalizeSpectrum, self).call(inputs, *args, **kwargs)
