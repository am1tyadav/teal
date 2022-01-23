from typing import Any
import tensorflow as tf
import numpy as np
from teal.augment.random_gain import RandomGain
from tests.utils import get_audio_examples
from tests.common import TealTest


class TestRandomGain(TealTest.TealTestCase):
    def setUp(self):
        self.power = 2
        self.setup_layer(
            layer=RandomGain(1.),
            single_example=get_audio_examples(1),
            batch_example=get_audio_examples(3)
        )

    def value_assertion(self, a: Any, b: Any):
        return self.assertAllClose(a, b, atol=0.01)

    def alternate_logic(self, inputs: tf.Tensor) -> np.ndarray:
        expected = []
        numpy_examples = inputs.numpy()
        num_examples = numpy_examples.shape[0]

        if num_examples == 1:
            results = self.single_result
        else:
            results = self.batch_result

        for i in range(0, num_examples):
            factor = (results[i] / (numpy_examples[i] + 1e-10)).numpy()
            factor = np.unique(factor.round(4)).max()
            expected.append(np.expand_dims(numpy_examples[i] * factor, axis=0))
        return np.concatenate(expected, axis=0)


if __name__ == "__main__":
    tf.test.main()
