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


class InversePolarity(AugmentationLayer):
    def __init__(self, chance: float):
        super(InversePolarity, self).__init__(chance=chance)

    def compute_augmentation(self, inputs):
        return -1. * inputs


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
            "_max_noise": self._max_noise
        })
        return config
