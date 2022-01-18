import tensorflow as tf
import numpy as np
import librosa
from teal.power_to_db import PowerToDb


class TestPowerToDb(tf.test.TestCase):
    def setUp(self):
        self._layer = PowerToDb()
        self._examples = np.random.normal(size=(4, 20, 30))
        self._results = self._layer(self._examples)
        self._expected_results = tf.constant(librosa.power_to_db(self._examples))

    def test_shapes(self):
        self.assertShapeEqual(self._results.numpy(), self._expected_results)

    def test_values(self):
        self.assertAllClose(self._results, self._expected_results,
                            rtol=0.05, atol=5.)


if __name__ == "__main__":
    tf.test.main()
