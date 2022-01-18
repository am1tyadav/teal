import tensorflow as tf
from tensorflow.keras import layers


class PowerToDb(layers.Layer):
    def __init__(self, epsilon: float = 1e-8):
        super(PowerToDb, self).__init__()

        self._epsilon = epsilon

    def call(self, inputs, *args, **kwargs):
        return tf.math.divide(
            tf.math.log(inputs + self._epsilon),
            tf.math.log(tf.constant(10, dtype=inputs.dtype))
        )

    def get_config(self):
        config = super(PowerToDb, self).get_config()
        config.update({
            "_epsilon": self._epsilon
        })
        return config