from typing import Any
from teal import AudioToMelSpectrogram
from tests.utils import *
from tests.common import TealTest


class TestMelSpectrogram(TealTest.TealTestCase):
    def setUp(self):
        self.power = 2.0
        self.setup_layer(
            layer=AudioToMelSpectrogram(
                SAMPLE_RATE, N_FFT, HOP_LEN, N_MELS, power=self.power
            ),
            single_example=get_audio_examples(1),
            batch_example=get_audio_examples(3),
            param_names=["_sample_rate", "_n_mels"],
        )

    def value_assertion(self, a: Any, b: Any):
        return self.assertAllClose(a, b, atol=30.0)

    def alternate_logic(self, inputs: tf.Tensor) -> np.ndarray:
        return from_audio_to_mel_spectrogram(inputs, self.power)


if __name__ == "__main__":
    tf.test.main()
