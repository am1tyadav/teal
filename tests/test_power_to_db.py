from typing import Any
import tensorflow as tf
import numpy as np
import librosa
from teal.power_to_db import PowerToDb
from tests.common import TealTest


class TestPowerToDb(TealTest.TealTestCase):
    def value_assertion(self, a: Any, b: Any):
        return self.assertAllClose(a, b, rtol=0.1, atol=0.1)

    def setUp(self):
        self.setup_layer(
            layer=PowerToDb(),
            single_example=tf.random.normal(shape=(1, 126, 128)),
            batch_example=tf.random.normal(shape=(3, 126, 128)),
            param_names=["_top_db", "_epsilon"]
        )

    def alternate_logic(self, inputs: tf.Tensor) -> np.ndarray:
        _numpy_examples = inputs.numpy()
        return librosa.power_to_db(_numpy_examples)


if __name__ == "__main__":
    tf.test.main()
