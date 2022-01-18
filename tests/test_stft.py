import tensorflow as tf
import numpy as np
import librosa
from teal.stft import STFT
from tests.utils import get_audio_examples, N_FFT, HOP_LEN


class TestSTFT(tf.test.TestCase):
    def setUp(self):
        self._layer = STFT(N_FFT, HOP_LEN)
        self._examples = get_audio_examples()
        self._results = self._layer(self._examples)

        _expected = []
        _numpy_examples = self._examples.numpy()
        _num_examples = _numpy_examples.shape[0]

        for i in range(0, _num_examples):
            _stft = librosa.stft(
                _numpy_examples[i], n_fft=N_FFT,
                hop_length=HOP_LEN, center=False
            )
            _stft = np.transpose(_stft)
            _expected.append(_stft)
        self._expected_results = tf.concat(np.array(_expected), axis=0)

    def test_shapes(self):
        self.assertShapeEqual(self._results.numpy(), self._expected_results)

    def test_values(self):
        self.assertAllClose(
            self._results, self._expected_results,
            atol=np.complex(0.1, 0.1), rtol=np.complex(0.01, 0.01)
        )


if __name__ == "__main__":
    tf.test.main()
