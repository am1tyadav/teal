"""AugmentationLayer

Base class to subclass data augmentation layers from
"""

from abc import abstractmethod
import tensorflow as tf
from tensorflow.keras import layers, backend

from teal.utils import load_audio


class AugmentationLayer(layers.Layer):
    """AugmentationLayer

    Base class to subclass data augmentation layers from
    """

    def __init__(self, chance: float, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert 0.0 < chance <= 1.0
        self._chance = chance

    @abstractmethod
    def compute_augmentation(self, inputs: tf.Tensor):
        """Abstract method that must be implemented by any data augmentation layers

        Args:
            inputs: tf.Tensor - Input tensor
        Returns:
            Augmented tensor
        """
        ...

    def call(self, inputs, *args, **kwargs):
        training = kwargs.get("training")

        if (
            tf.random.uniform(shape=(), dtype=tf.float32, minval=0.0, maxval=1.0)
            < self._chance
        ):
            augmented = self.compute_augmentation(inputs)
        else:
            augmented = inputs

        return backend.in_train_phase(augmented, inputs, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({"_chance": self._chance})
        return config


class InversePolarity(AugmentationLayer):
    """InversePolarity

    Invert the polarity of input audio
    """

    def __init__(self, chance: float, *args, **kwargs):
        super().__init__(chance=chance, *args, **kwargs)

    def compute_augmentation(self, inputs):
        return inputs * -1.0


class NoiseBank(AugmentationLayer):
    """NoiseBank

    Applies noise from noise bank to input audio
    """

    def __init__(
        self,
        chance: float,
        noise_source: str,
        *args,
        samples_len: int = 66150,
        noise_weight: float = 0.2,
        sample_rate: int = 22050,
        **kwargs
    ):
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
            return _example + self._noise[_index : _index + self._samples_len]

        batch_size = tf.shape(inputs)[0]
        starting_sample = tf.random.uniform(
            shape=(batch_size,),
            minval=0,
            maxval=self._duration - self._samples_len,
            dtype=tf.int64,
        )
        starting_sample = tf.expand_dims(
            tf.cast(starting_sample, dtype=tf.float32), axis=-1
        )
        _examples = tf.concat(values=[inputs, starting_sample], axis=1)
        outputs = tf.map_fn(_add_noise, _examples)
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "_samples_len": self._samples_len,
                "_noise_source": self._noise_source,
                "_noise_weight": self._noise_weight,
                "_sample_rate": self._sample_rate,
            }
        )
        return config


class RandomGain(AugmentationLayer):
    """RandomGain

    Applies random gain to input audio
    """

    def __init__(
        self,
        chance: float,
        *args,
        min_factor: float = 0.5,
        max_factor: float = 0.9,
        **kwargs
    ):
        super().__init__(chance, *args, **kwargs)

        self._min_factor = min_factor
        self._max_factor = max_factor

    def compute_augmentation(self, inputs):
        batch_size = tf.shape(inputs)[0]
        gain_factors = tf.random.uniform(
            (batch_size,),
            minval=self._min_factor,
            maxval=self._max_factor,
            dtype=inputs.dtype,
        )
        return tf.math.multiply(inputs, tf.expand_dims(gain_factors, axis=1))

    def get_config(self):
        config = super().get_config()
        config.update(
            {"_min_factor": self._min_factor, "_max_factor": self._max_factor}
        )
        return config


class RandomNoise(AugmentationLayer):
    """RandomNoise

    Applies random noise to input audio
    """

    def __init__(self, chance: float, *args, max_noise: float = 0.01, **kwargs):
        super().__init__(chance, *args, **kwargs)

        self._max_noise = max_noise

    def compute_augmentation(self, inputs):
        random_noise = tf.random.uniform(
            tf.shape(inputs),
            minval=-1.0 * self._max_noise,
            maxval=self._max_noise,
            dtype=inputs.dtype,
        )
        return inputs + random_noise

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "_max_noise": self._max_noise,
            }
        )
        return config


class PitchShift(AugmentationLayer):
    """PitchShift

    Basic pitch shifter which computes fft, shifts and ifft
    """

    def __init__(self, chance: float, shift: int, *args, **kwargs):
        super().__init__(chance=chance, *args, **kwargs)

        self._shift = shift

    def compute_augmentation(self, inputs):
        def _pitch_shift(single_audio):
            _shift = tf.random.uniform(
                shape=(), minval=-self._shift, maxval=self._shift, dtype=tf.int64
            )

            r_fft = tf.signal.rfft(single_audio)
            r_fft = tf.roll(r_fft, _shift, axis=0)
            zeros = tf.complex(tf.zeros([tf.abs(_shift)]), tf.zeros([tf.abs(_shift)]))

            if _shift < 0:
                r_fft = tf.concat([r_fft[:_shift], zeros], axis=0)
            else:
                r_fft = tf.concat([zeros, r_fft[_shift:]], axis=0)
            return tf.signal.irfft(r_fft)

        return tf.map_fn(_pitch_shift, inputs)

    def get_config(self):
        config = super().get_config()
        config.update({"_shift": self._shift})
        return config
