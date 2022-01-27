"""RandomNoise

Applies random noise to input audio
"""

import tensorflow as tf
from teal import AugmentationLayer


class RandomNoise(AugmentationLayer):
    """RandomNoise

    Applies random noise to input audio
    """
    def __init__(self,
                 chance: float,
                 *args,
                 max_noise: float = 0.01,
                 **kwargs):
        super().__init__(chance, *args, **kwargs)

        self._max_noise = max_noise

    def compute_augmentation(self, inputs):
        random_noise = tf.random.uniform(
            tf.shape(inputs),
            minval=-1. * self._max_noise,
            maxval=self._max_noise,
            dtype=inputs.dtype
        )
        return inputs + random_noise

    def get_config(self):
        config = super().get_config()
        config.update({
            "_max_noise": self._max_noise,
        })
        return config
