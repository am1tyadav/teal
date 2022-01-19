"""Normalization Layers

Normalize
    NormalizeAudio
    NormalizeSpectrum
"""

from typing import Any
import tensorflow as tf
from tensorflow.keras import layers


class Normalize(layers.Layer):
    def __init__(self,
                 axes: Any,
                 *args,
                 expand: int = 1,
                 epsilon: float = 1e-10,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self._axes = axes
        self._expand = expand
        self._epsilon = epsilon

    def call(self, inputs, *args, **kwargs):
        _max = tf.reduce_max(tf.abs(inputs), axis=self._axes)

        for _ in range(0, self._expand):
            _max = tf.expand_dims(_max + self._epsilon, axis=-1)
        return inputs / _max

    def get_config(self):
        config = super().get_config()
        config.update({
            "_axes": self._axes,
            "_expand": self._expand,
            "_epsilon": self._epsilon
        })


class NormalizeAudio(Normalize):
    def __init__(self, *args, **kwargs):
        super().__init__(1, *args, expand=1, **kwargs)


class NormalizeSpectrum(Normalize):
    def __init__(self, *args, **kwargs):
        super().__init__((1, 2), *args, expand=2, **kwargs)
