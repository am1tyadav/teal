# import tensorflow as tf
# import numpy as np
# import librosa
# from teal.mel_spectrogram import MelSpectrogram
# from tests.utils import get_audio_examples, N_FFT, HOP_LEN, SAMPLE_RATE, N_MELS
#
#
# class TestMelSpectrogram(tf.test.TestCase):
#     def setUp(self):
#         _power = 1.
#         self._layer = MelSpectrogram(
#             sample_rate=SAMPLE_RATE, n_fft=N_FFT,
#             n_mels=N_MELS, hop_length=HOP_LEN, power=_power
#         )
#         self._examples = get_audio_examples()
#         self._results = self._layer(self._examples)
#
#         _expected = []
#         _numpy_examples = self._examples.numpy()
#         _num_examples = _numpy_examples.shape[0]
#
#         for i in range(0, _num_examples):
#             _mel_spec = librosa.feature.melspectrogram(
#                 y=_numpy_examples[i], sr=SAMPLE_RATE,
#                 n_fft=N_FFT, hop_length=HOP_LEN, n_mels=N_MELS,
#                 center=False, power=_power, fmax=SAMPLE_RATE/2
#             )
#             _mel_spec = np.transpose(_mel_spec)
#             _expected.append(_mel_spec)
#         self._expected_results = tf.concat(np.array(_expected), axis=0)
#
#     def test_shapes(self):
#         self.assertShapeEqual(self._results.numpy(), self._expected_results)
#
#     def test_values(self):
#         self.assertAllClose(
#             self._results, self._expected_results,
#             atol=0.1, rtol=0.01
#         )
#
#
# if __name__ == "__main__":
#     tf.test.main()
