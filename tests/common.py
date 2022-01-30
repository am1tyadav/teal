from abc import abstractmethod
from typing import List, Any
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer


class TealTest:
    class TealTestCase(tf.test.TestCase):
        def setup_layer(
            self,
            layer: Layer,
            single_example: tf.Tensor,
            batch_example: tf.Tensor,
            param_names: List[str] = None,
        ):
            self.layer = layer
            self.config = self.layer.get_config()
            self.param_names = param_names if param_names is not None else []
            self.single_example = single_example
            self.batch_example = batch_example
            self.single_result = self.layer(self.single_example, training=True)
            self.batch_result = self.layer(self.batch_example, training=True)

        def test_config_is_not_none(self):
            self.assertNotEqual(None, self.config)

        def test_params_exist_in_config(self):
            for name in self.param_names:
                self.assertIn(name, self.config.keys())

        @abstractmethod
        def alternate_logic(self, inputs: tf.Tensor) -> np.ndarray:
            ...

        def value_assertion(self, a: Any, b: Any):
            return self.assertAllEqual(a, b)

        def test_single_example(self):
            _expected = self.alternate_logic(self.single_example)

            self.assertShapeEqual(_expected, self.single_result)
            self.value_assertion(_expected, self.single_result)

        def test_batch_example(self):
            _expected = self.alternate_logic(self.batch_example)

            self.assertShapeEqual(_expected, self.batch_result)
            self.value_assertion(_expected, self.batch_result)
