from abc import abstractmethod
import tensorflow as tf
from tensorflow.keras import layers, backend


class AugmentationLayer(layers.Layer):
    def __init__(self, chance: float):
        super(AugmentationLayer, self).__init__()

        assert 0. < chance <= 1.
        self._chance = chance

    @abstractmethod
    def compute_augmentation(self, inputs):
        ...

    def call(self, inputs, *args, **kwargs):
        training = kwargs.get("training")

        if tf.random.uniform(shape=(), minval=0., maxval=1.) < self._chance:
            augmented = self.compute_augmentation(inputs)
        else:
            augmented = inputs

        return backend.in_train_phase(augmented, inputs, training=training)

    def get_config(self):
        config = super(AugmentationLayer, self).get_config()
        config.update({
            "_chance": self._chance
        })
        return config


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
