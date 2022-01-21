from typing import Any
import tensorflow as tf
import numpy as np
import librosa
from teal.power_to_db import PowerToDb
from tests.common import TealTest
from tests.utils import get_spectrogram_examples


class TestPowerToDb(TealTest.TealTestCase):
    def value_assertion(self, a: Any, b: Any):
        return self.assertAllClose(a, b, atol=10.)

    def setUp(self):
        self.power = 2
        self.setup_layer(
            layer=PowerToDb(),
            single_example=get_spectrogram_examples(1, self.power),
            batch_example=get_spectrogram_examples(3, self.power),
            param_names=["_top_db", "_epsilon"]
        )

    def alternate_logic(self, inputs: tf.Tensor) -> np.ndarray:
        _numpy_examples = inputs.numpy()
        return librosa.power_to_db(_numpy_examples)


if __name__ == "__main__":
    tf.test.main()
