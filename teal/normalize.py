"""Normalization Layers

Normalize
    NormalizeAudio
    NormalizeSpectrum
"""

from typing import Any
import tensorflow as tf
from tensorflow.keras import layers


class Normalize(layers.Layer):
    """Normalizes input tensor to a range of -1, 1

    The input tensor can be of rank 2 or 3
    """
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
        return config


class NormalizeAudio(Normalize):
    """Normalizes input tensor of rank 2 to a range of (-1, 1)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(1, *args, expand=1, **kwargs)


class NormalizeSpectrum(Normalize):
    """Normalizes input tensor of rank 3 to a range of (-1, 1)
    """
    def __init__(self, *args, **kwargs):
        super().__init__((1, 2), *args, expand=2, **kwargs)
