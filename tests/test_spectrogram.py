from typing import Any
import tensorflow as tf
import numpy as np
import librosa
from teal.spectrogram import Spectrogram
from tests.utils import get_audio_examples, N_FFT, HOP_LEN
from tests.common import TealTest


# Todo - input needs to be complex numbers


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
        _expected = []
        _numpy_examples = inputs.numpy()
        _num_examples = _numpy_examples.shape[0]

        for i in range(0, _num_examples):
            _spec, _ = librosa.core.spectrum._spectrogram(
                y=_numpy_examples[i],
                n_fft=N_FFT, hop_length=HOP_LEN,
                power=self.power, center=False
            )
            _spec = np.expand_dims(np.transpose(_spec), axis=0)
            _expected.append(_spec)

        return np.concatenate(_expected, axis=0)


if __name__ == "__main__":
    tf.test.main()
