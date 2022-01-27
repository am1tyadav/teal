import tensorflow as tf
import numpy as np
from teal import NormalizeAudio, NormalizeSpectrum
from tests.common import TealTest
from tests.utils import get_audio_examples, get_spectrogram_examples


class TestNormalizeAudio(TealTest.TealTestCase):
    def setUp(self):
        self.setup_layer(
            NormalizeAudio(),
            single_example=get_audio_examples(1),
            batch_example=get_audio_examples(3),
            param_names=["_axes", "_expand", "_epsilon"]
        )

    def alternate_logic(self, inputs: tf.Tensor) -> np.ndarray:
        _numpy_example = inputs.numpy()
        _max = np.max(np.abs(_numpy_example), axis=1) + 1e-10
        _max = np.expand_dims(_max, axis=1)
        _norm = _numpy_example / _max
        return _norm


class TestNormalizeSpectrum(TealTest.TealTestCase):
    def setUp(self):
        self.power = 2
        self.setup_layer(
            layer=NormalizeSpectrum(),
            single_example=get_spectrogram_examples(1, self.power),
            batch_example=get_spectrogram_examples(3, self.power),
            param_names=["_axes", "_expand", "_epsilon"]
        )

    def alternate_logic(self, inputs: tf.Tensor) -> np.ndarray:
        _numpy_example = inputs.numpy()
        _abs = np.abs(_numpy_example)
        _max = np.max(_abs, axis=-1) + 1e-10
        _max = np.max(_max, axis=-1) + 1e-10
        _max = np.expand_dims(_max, axis=-1)
        _max = np.expand_dims(_max, axis=-1)
        _norm = _numpy_example / _max
        return _norm


if __name__ == "__main__":
    tf.test.main()
