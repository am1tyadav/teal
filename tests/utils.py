import os
import random
import tensorflow as tf
import numpy as np


FILE_PATH = os.path.join(os.getcwd(), "samples", "led.wav")
SAMPLE_RATE = 22050
N_FFT = 2048
N_MELS = 128
HOP_LEN = 512
DURATION = 29
CHUNK_LENGTH = 3


def get_audio_examples(num_examples: int = 2):
    def _get_chunk(y):
        start_sample = random.randint(0, DURATION - CHUNK_LENGTH) * SAMPLE_RATE
        end_sample = start_sample + CHUNK_LENGTH * SAMPLE_RATE
        return y[start_sample: end_sample]

    with open(FILE_PATH, "rb") as f:
        audio_data = f.read()

    audio, _ = tf.audio.decode_wav(audio_data)
    audio = tf.transpose(audio).numpy()[0]
    examples = np.zeros((num_examples, CHUNK_LENGTH * SAMPLE_RATE))

    for i in range(0, num_examples):
        examples[i] = _get_chunk(audio)

    assert examples.shape == (num_examples, SAMPLE_RATE * CHUNK_LENGTH), "audio_chunk shape is wrong"
    return tf.constant(examples, dtype=tf.float32)
