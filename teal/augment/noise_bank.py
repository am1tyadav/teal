"""NoiseBank

Applies noise from noise bank to input audio
"""

import tensorflow as tf
from teal.augment.augment import AugmentationLayer
from teal.utils import load_audio


class NoiseBank(AugmentationLayer):
    """NoiseBank

    Applies noise from noise bank to input audio
    """
    def __init__(self,
                 chance: float,
                 noise_source: str,
                 *args,
                 samples_len: int = 66150,
                 noise_weight: float = 0.2,
                 sample_rate: int = 22050,
                 **kwargs):
        """Apply randomly selected noise from a given file to input tensor

        User given wav files are loaded when the layer is built to be applied
        as noises when input is passed to the layer

        It is expected that the given samples_len will match with the input dimension

        Args:
            chance: float - Likelihood of augmentation applied in each call
            samples_len: int - Number of samples to use from the noise audio
            noise_source: str - File path of audio file to use as noise
            noise_weight: float - Weights assigned to the noise
            sample_rate: int - Expected sample rate of the audio file
        """
        super().__init__(chance, *args, **kwargs)

        self._samples_len = samples_len
        self._noise_source = noise_source
        self._noise_weight = noise_weight
        self._sample_rate = sample_rate
        self._noise = None
        self._duration = None

    def build(self, input_shape):
        self._noise = load_audio(self._noise_source, self._sample_rate)
        self._duration = tf.cast(tf.shape(self._noise)[0], dtype=tf.int64)
        assert self._duration >= self._samples_len, "Audio file is too short"
        assert self._samples_len == input_shape[1], "Input shape mismatch"

    def compute_augmentation(self, inputs):
        batch_size = tf.shape(inputs)[0]
        starting_sample = tf.random.uniform(
            shape=(batch_size,),
            minval=0,
            maxval=self._duration - self._samples_len,
            dtype=tf.int64
        )
        outputs = None

        for i in range(0, batch_size):
            start_index = starting_sample[i]
            out = self._noise_weight * self._noise[start_index: start_index + self._samples_len]
            out = out + (1 - self._noise_weight) * inputs[i]
            out = tf.expand_dims(out, axis=0)

            if outputs is None:
                outputs = out
            else:
                outputs = tf.concat([outputs, out], axis=0)
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({
            "_samples_len": self._samples_len,
            "_noise_source": self._noise_source,
            "_noise_weight": self._noise_weight,
            "_sample_rate": self._sample_rate
        })
        return config
