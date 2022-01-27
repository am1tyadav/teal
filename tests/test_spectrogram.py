from typing import Any
import tensorflow as tf
import numpy as np
from teal import Spectrogram
from tests.utils import from_audio_to_spectrogram, get_audio_examples, N_FFT, HOP_LEN
from tests.common import TealTest


class TestSpectrogram(TealTest.TealTestCase):
    def setUp(self):
        self.power = 2
        self.setup_layer(
            layer=Spectrogram(N_FFT, HOP_LEN, power=self.power),
            single_example=get_audio_examples(1),
            batch_example=get_audio_examples(3)
        )

    def value_assertion(self, a: Any, b: Any):
        return self.assertAllClose(a, b, atol=0.01, rtol=0.01)

    def alternate_logic(self, inputs: tf.Tensor) -> np.ndarray:
        return from_audio_to_spectrogram(inputs, self.power)


if __name__ == "__main__":
    tf.test.main()
