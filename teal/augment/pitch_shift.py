"""PitchShift

Basic pitch shifter which computes fft, shifts and inverse fft
"""

import tensorflow as tf
from teal.augment.augment import AugmentationLayer


class PitchShift(AugmentationLayer):
    """PitchShift

    Basic pitch shifter which computes fft, shifts and ifft
    """

    def __init__(self,
                 chance: float,
                 shift: int,
                 *args,
                 **kwargs):
        super().__init__(chance=chance, *args, **kwargs)

        self._shift = shift

    def compute_augmentation(self, inputs):
        def _pitch_shift(single_audio):
            _shift = tf.random.uniform(
                shape=(),
                minval=-self._shift,
                maxval=self._shift,
                dtype=tf.int64
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
