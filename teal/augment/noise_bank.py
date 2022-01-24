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
        def _add_noise(example):
            _example = example[:-1]
            _index = tf.cast(example[-1], dtype=tf.int64)
            return _example + self._noise[_index: _index + self._samples_len]

        batch_size = tf.shape(inputs)[0]
        starting_sample = tf.random.uniform(
            shape=(batch_size,),
            minval=0,
            maxval=self._duration - self._samples_len,
            dtype=tf.int64
        )
        starting_sample = tf.expand_dims(tf.cast(starting_sample, dtype=tf.float32), axis=-1)
        _examples = tf.concat(values=[inputs, starting_sample], axis=1)
        outputs = tf.map_fn(_add_noise, _examples)
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
