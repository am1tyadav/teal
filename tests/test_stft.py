from typing import Any
import tensorflow as tf
import numpy as np
import librosa
from teal.stft import STFT
from tests.utils import get_audio_examples, N_FFT, HOP_LEN
from tests.common import TealTest


class TestSTFT(TealTest.TealTestCase):
    def setUp(self):
        self.setup_layer(
            layer=STFT(N_FFT, HOP_LEN),
            single_example=get_audio_examples(1),
            batch_example=get_audio_examples(3),
            param_names=["_n_fft", "_hop_length"]
        )

    def value_assertion(self, a: Any, b: Any):
        return self.assertAllClose(
            a, b,
            atol=np.complex(0.1, 0.1),
            rtol=np.complex(0.01, 0.01)
        )

    def alternate_logic(self, inputs: tf.Tensor) -> np.ndarray:
        _expected = []
        _numpy_examples = inputs.numpy()
        _num_examples = _numpy_examples.shape[0]

        for i in range(0, _num_examples):
            _stft = librosa.stft(
                _numpy_examples[i], n_fft=N_FFT,
                hop_length=HOP_LEN, center=False
            )
            _stft = np.expand_dims(np.transpose(_stft), axis=0)
            _expected.append(_stft)

        return np.concatenate(_expected, axis=0)


if __name__ == "__main__":
    tf.test.main()
