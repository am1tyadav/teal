from typing import Any
import tensorflow as tf
import numpy as np
from teal import AudioToSTFT
from tests.utils import get_audio_examples, from_audio_to_stft, N_FFT, HOP_LEN
from tests.common import TealTest


class TestSTFT(TealTest.TealTestCase):
    def setUp(self):
        self.setup_layer(
            layer=AudioToSTFT(N_FFT, HOP_LEN),
            single_example=get_audio_examples(1),
            batch_example=get_audio_examples(3),
            param_names=["_n_fft", "_hop_length"],
        )

    def value_assertion(self, a: Any, b: Any):
        return self.assertAllClose(
            a, b, atol=np.complex(0.1, 0.1), rtol=np.complex(0.01, 0.01)
        )

    def alternate_logic(self, inputs: tf.Tensor) -> np.ndarray:
        return from_audio_to_stft(inputs)


if __name__ == "__main__":
    tf.test.main()
