import tensorflow as tf
from teal.inverse_polarity import InversePolarity
from tests.utils import get_audio_examples


class TestInversePolarity(tf.test.TestCase):
    def setUp(self):
        self._layer = InversePolarity(chance=1.)
        self._examples = get_audio_examples()
        self._results_train = self._layer(self._examples, training=True)
        self._results_eval = self._layer(self._examples, training=False)
        self._silences = self._results_train + self._results_eval

    def test_shapes(self):
        self.assertShapeEqual(self._results_train.numpy(), self._examples)
        self.assertShapeEqual(self._results_eval.numpy(), self._examples)

    def test_values(self):
        self.assertAllEqual(self._silences, tf.zeros_like(self._silences))


if __name__ == "__main__":
    tf.test.main()
