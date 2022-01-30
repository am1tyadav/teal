import os
import random
import tensorflow as tf
import librosa
import numpy as np
from teal.utils import load_audio


FILE_PATH = os.path.join(os.getcwd(), "samples", "led.wav")
SAMPLE_RATE = 22050
N_FFT = 2048
N_MELS = 128
HOP_LEN = 512
DURATION = 29
CHUNK_LENGTH = 3
NUM_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE


def get_audio_examples(num_examples: int = 2):
    def _get_chunk(y):
        start_sample = random.randint(0, DURATION - CHUNK_LENGTH) * SAMPLE_RATE
        end_sample = start_sample + NUM_SAMPLES
        return y[start_sample:end_sample]

    audio = load_audio(file_path=FILE_PATH, expected_sr=SAMPLE_RATE)
    audio = audio.numpy()

    examples = np.zeros((num_examples, NUM_SAMPLES))

    for i in range(0, num_examples):
        examples[i] = _get_chunk(audio)

    assert examples.shape == (num_examples, NUM_SAMPLES), "audio_chunk shape is wrong"
    return tf.constant(examples, dtype=tf.float32)


def from_audio_to_stft(inputs: tf.Tensor):
    _expected = []
    _numpy_examples = inputs.numpy()
    _num_examples = _numpy_examples.shape[0]

    for i in range(0, _num_examples):
        _stft = librosa.stft(
            _numpy_examples[i], n_fft=N_FFT, hop_length=HOP_LEN, center=False
        )
        _stft = np.expand_dims(np.transpose(_stft), axis=0)
        _expected.append(_stft)

    return np.concatenate(_expected, axis=0)


def get_stft_examples(num_examples: int = 2):
    _examples = get_audio_examples(num_examples)
    return tf.constant(from_audio_to_stft(_examples))


def from_audio_to_spectrogram(inputs: tf.Tensor, power: float):
    _expected = []
    _numpy_examples = inputs.numpy()
    _num_examples = _numpy_examples.shape[0]

    for i in range(0, _num_examples):
        _spec, _ = librosa.core.spectrum._spectrogram(
            y=_numpy_examples[i],
            n_fft=N_FFT,
            hop_length=HOP_LEN,
            power=power,
            center=False,
        )
        _spec = np.expand_dims(np.transpose(_spec), axis=0)
        _expected.append(_spec)

    return np.concatenate(_expected, axis=0)


def get_spectrogram_examples(num_examples: int = 2, power: float = 2.0):
    _examples = get_audio_examples(num_examples)
    return tf.constant(from_audio_to_spectrogram(_examples, power=power))


def from_audio_to_mel_spectrogram(inputs: tf.Tensor, power: float):
    _expected = []

    _specs = from_audio_to_spectrogram(inputs, power)
    _filter_bank = librosa.filters.mel(
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        norm=None,
        fmin=0,
        fmax=SAMPLE_RATE / 2,
        htk=True,
    )
    _filter_bank = np.transpose(_filter_bank)

    _numpy_examples = inputs.numpy()
    _num_examples = _numpy_examples.shape[0]

    for i in range(0, _num_examples):
        _mel_spec = np.dot(_specs[i], _filter_bank)
        _mel_spec = np.expand_dims(_mel_spec, axis=0)
        _expected.append(_mel_spec)
    return np.concatenate(_expected, axis=0)
