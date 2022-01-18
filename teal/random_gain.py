import tensorflow as tf
from teal.augment import AugmentationLayer


class RandomGain(AugmentationLayer):
    def __init__(self, chance: float,
                 min_factor: float = 0.5,
                 max_factor: float = 0.9):
        super(RandomGain, self).__init__(chance=chance)

        self._min_factor = min_factor
        self._max_factor = max_factor

    def compute_augmentation(self, inputs):
        batch_size = tf.shape(inputs)[0]
        gain_factors = tf.random.uniform(
            (batch_size,),
            minval=self._min_factor,
            maxval=self._max_factor,
            dtype=inputs.dtype
        )
        return tf.math.multiply(inputs, tf.expand_dims(gain_factors, axis=1))

    def get_config(self):
        config = super(RandomGain, self).get_config()
        config.update({
            "_min_factor": self._min_factor,
            "_max_factor": self._max_factor,
        })
        return config
