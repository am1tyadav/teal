"""NoiseBank

Applies noise from noise bank to input audio
"""

from typing import List
import tensorflow as tf
from teal.augment import AugmentationLayer
from teal.utils import load_audio


# Todo - Provide a set of default noise files


class NoiseBank(AugmentationLayer):
    def __init__(self,
                 chance: float,
                 num_samples: int,
                 noise_bank: List[str],
                 noise_weights: List[float],
                 sample_rate: int,
                 *args,
                 **kwargs):
        """
        :param chance: Likelihood of augmentation applied in each call
        :param num_samples: Number of samples to use from the noise audio
        :param noise_bank: File paths of audio files to use as noises
        :param noise_weights: Weights assigned to the noise_bank files
        :param sample_rate: Expected sample rate of the audio files
        """
        super().__init__(chance, *args, **kwargs)

        self._num_samples = num_samples
        self._noise_bank: List[str] = noise_bank
        self._noise_weights = noise_weights
        self._sample_rate = sample_rate
        self._noises = None

    def build(self, input_shape):
        # Load noise bank files as tensors
        for file_path in self._noise_bank:
            noise_sample = load_audio(file_path, self._sample_rate)[:self._num_samples]
            noise_sample = tf.expand_dims(noise_sample, axis=0)

            if self._noises is None:
                self._noises = noise_sample
            else:
                self._noises = tf.concat([self._noises, noise_sample], axis=0)

    def compute_augmentation(self, inputs):
        batch_size = tf.shape(inputs)[0]
        noise_indices = tf.random.uniform(
            shape=(batch_size,),
            minval=0,
            maxval=len(self._noise_bank),
            dtype=tf.int64
        )
        outputs = None

        for i in range(0, batch_size):
            noise_index = noise_indices[i]
            out = self._noise_weights[noise_index] * self._noises[noise_index] + \
                   (1 - self._noise_weights[noise_index]) * inputs[i]
            out = tf.expand_dims(out, axis=0)

            if outputs is None:
                outputs = out
            else:
                outputs = tf.concat([outputs, out], axis=0)
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({
            "_num_samples": self._num_samples,
            "_noise_bank": self._noise_bank,
            "_noise_weights": self._noise_weights,
            "_sample_rate": self._sample_rate
        })
        return config
