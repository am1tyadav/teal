"""PowerToDb

Scales input spectrum power to db
"""

import tensorflow as tf
from tensorflow.keras import layers


class PowerToDb(layers.Layer):
    """PowerToDb

    Scales input spectrum power to db
    """

    def __init__(self, *args, top_db: float = 110.0, epsilon: float = 1e-10, **kwargs):
        super().__init__(*args, **kwargs)

        self._top_db = top_db
        self._epsilon = epsilon

    def call(self, inputs, *args, **kwargs):
        _epsilon = tf.zeros_like(inputs) + self._epsilon
        _log = tf.math.log(tf.maximum(inputs, _epsilon))
        _log_10 = 10 * tf.math.divide(
            _log, tf.math.log(tf.constant(10, dtype=inputs.dtype))
        )
        _value_max = tf.expand_dims(
            tf.reduce_max(_log_10, axis=(1, 2)) - self._top_db, axis=-1
        )
        _value_max = tf.expand_dims(_value_max, axis=-1)
        _clipped = tf.maximum(_log_10, _value_max)
        return _clipped

    def get_config(self):
        config = super().get_config()
        config.update({"_top_db": self._top_db, "_epsilon": self._epsilon})
        return config
