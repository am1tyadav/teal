import tensorflow as tf
import numpy as np
import librosa
from teal.spectrogram import Spectrogram
from tests.utils import get_audio_examples, N_FFT, HOP_LEN


class TestSpectrogram(tf.test.TestCase):
    def setUp(self):
        self._layer = Spectrogram(N_FFT, HOP_LEN)
        self._examples = get_audio_examples()
        self._results = self._layer(self._examples)

        _expected = []
        _numpy_examples = self._examples.numpy()
        _num_examples = _numpy_examples.shape[0]

        for i in range(0, _num_examples):
            _spec, _ = librosa.core.spectrum._spectrogram(
                y=_numpy_examples[i],
                n_fft=N_FFT, hop_length=HOP_LEN,
                power=2, center=False
            )
            _spec = np.transpose(_spec)
            _expected.append(_spec)
        self._expected_results = tf.concat(np.array(_expected), axis=0)

    def test_shapes(self):
        self.assertShapeEqual(self._results.numpy(), self._expected_results)

    def test_values(self):
        self.assertAllClose(
            self._results, self._expected_results,
            atol=0.1, rtol=0.01
        )


if __name__ == "__main__":
    tf.test.main()
