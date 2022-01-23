"""NoiseBank

Applies noise from noise bank to input audio
"""

from typing import List
import tensorflow as tf
from teal.augment import AugmentationLayer
from teal.utils import load_audio


class NoiseBank(AugmentationLayer):
    """NoiseBank

    Applies noise from noise bank to input audio
    """
    def __init__(self,
                 chance: float,
                 num_samples: int,
                 noise_bank: List[str],
                 noise_weights: List[float],
                 sample_rate: int,
                 *args,
                 **kwargs):
        """Apply randomly selected noises to input tensor

        User given wav files are loaded when the layer is built to be applied
        as noises when input is passed to the layer

        Args:
            chance: float - Likelihood of augmentation applied in each call
            num_samples: int - Number of samples to use from the noise audio
            noise_bank: List[str] - File paths of audio files to use as noises
            noise_weights: List[float] - Weights assigned to the noise_bank files
            sample_rate: int - Expected sample rate of the audio files
        """
        super().__init__(chance, *args, **kwargs)

        self._num_samples = num_samples
        self._noise_bank: List[str] = noise_bank
        self._noise_weights = noise_weights
        self._sample_rate = sample_rate
        self._noises = None

    def build(self, input_shape):
        for file_path in self._noise_bank:
            noise_sample = load_audio(file_path, self._sample_rate)

            assert tf.shape(noise_sample)[0] >= self._num_samples, "Audio file is too short"

            noise_sample = noise_sample[:self._num_samples]
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
