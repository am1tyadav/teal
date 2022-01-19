"""AugmentationLayer

Base class to subclass data augmentation layers from
"""

from abc import abstractmethod
import tensorflow as tf
from tensorflow.keras import layers, backend


class AugmentationLayer(layers.Layer):
    """AugmentationLayer

    Base class to subclass data augmentation layers from
    """
    def __init__(self,
                 chance: float,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        assert 0. < chance <= 1.
        self._chance = chance

    @abstractmethod
    def compute_augmentation(self, inputs):
        """compute_augmentation

        This function should be implemented by subclassed layers
        With the data augmentation logic

        :param inputs: Input tensor
        :return: Augmented tensor
        """
        ...

    def call(self, inputs, *args, **kwargs):
        training = kwargs.get("training")

        if tf.random.uniform(
                shape=(),
                dtype=tf.float32,
                minval=0.,
                maxval=1.) < self._chance:
            augmented = self.compute_augmentation(inputs)
        else:
            augmented = inputs

        return backend.in_train_phase(augmented, inputs, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            "_chance": self._chance
        })
        return config
