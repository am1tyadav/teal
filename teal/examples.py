import tensorflow as tf
import teal


def create_log_mel_spectrogram_model(
        num_samples: int = 66150,
        sample_rate: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128) -> tf.keras.models.Model:
    _model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(num_samples, )),
        teal.NormalizeAudio(),
        teal.MelSpectrogram(sample_rate, n_fft, hop_length, n_mels),
        teal.PowerToDb(),
        teal.NormalizeSpectrum()
    ])
    return _model


def create_data_augmentation_model(num_samples: int = 66150) -> tf.keras.models.Model:
    _model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(num_samples, )),
        teal.InversePolarity(0.5),
        teal.RandomGain(0.5),
        teal.RandomNoise(0.5),
        teal.RandomGain(0.5)
    ])
    return _model
