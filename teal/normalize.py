import tensorflow as tf
from tensorflow.keras import layers


class Normalize(layers.Layer):
    def __init__(self, epsilon: float = 1e-8):
        super(Normalize, self).__init__()

        self._epsilon = epsilon

    def call(self, inputs, *args, **kwargs):
        _abs = tf.abs(inputs)
        _max = tf.reduce_max(_abs, axis=1)
        _normalized = inputs / (tf.expand_dims(_max, axis=1) + self._epsilon)
        return _normalized

    def get_config(self):
        config = super(Normalize, self).get_config()
        config.update({
            "_epsilon": self._epsilon
        })
        return config
