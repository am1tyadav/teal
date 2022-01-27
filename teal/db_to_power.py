"""DbToPower

Scales Db to Power
"""

import tensorflow as tf
from tensorflow.keras import layers


class DbToPower(layers.Layer):
    """DbToPower

    Scales input db magnitude to power
    """

    def __init__(self,
                 *args,
                 ref: float = 1.,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self._ref = ref

    def call(self, inputs, *args, **kwargs):
        _scaled = inputs / 10.
        _pow = tf.pow(10., _scaled)
        return self._ref * _pow

    def get_config(self):
        config = super().get_config()
        config.update({
            "_ref": self._ref
        })
        return config
