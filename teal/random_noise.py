import tensorflow as tf
from teal.augment import AugmentationLayer


class RandomNoise(AugmentationLayer):
    def __init__(self, chance: float,
                 max_noise: float = 0.05):
        super(RandomNoise, self).__init__(chance=chance)

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
        config = super(RandomNoise, self).get_config()
        config.update({
            "_max_noise": self._max_noise,
        })
        return config
