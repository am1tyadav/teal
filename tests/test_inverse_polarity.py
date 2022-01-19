import numpy as np
import tensorflow as tf
from teal.inverse_polarity import InversePolarity
from tests.common import TealTest
from tests.utils import get_audio_examples


class TestInversePolarity(TealTest.TealTestCase):
    def setUp(self):
        self.setup_layer(
            InversePolarity(1.),
            single_example=get_audio_examples(1),
            batch_example=get_audio_examples(3),
        )

    def alternate_logic(self, inputs: tf.Tensor) -> np.ndarray:
        _numpy_examples = inputs.numpy()
        return -1. * _numpy_examples

    def test_single_example(self):
        _expected = self.alternate_logic(self.single_example)
        _silence = _expected + self.single_example.numpy()

        self.assertShapeEqual(_expected, self.single_result)
        self.assertAllEqual(_expected, self.single_result)
        self.assertAllEqual(_silence, np.zeros_like(_silence))

    def test_batch_example(self):
        _expected = self.alternate_logic(self.batch_example)
        _silence = _expected + self.batch_example.numpy()

        self.assertShapeEqual(_expected, self.batch_result)
        self.assertAllEqual(_expected, self.batch_result)
        self.assertAllEqual(_silence, np.zeros_like(_silence))


if __name__ == "__main__":
    tf.test.main()
