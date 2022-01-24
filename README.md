# Teal

__Teal__ is a library of TensorFlow layers written for audio data preprocessing

Easily create TensorFlow models for audio preprocessing and audio data augmentation:

:heavy_check_mark: No dependency other than TensorFlow

:heavy_check_mark: Can utilize GPU

:heavy_check_mark: Online preprocessing and data augmentation

:heavy_check_mark: Deploy preprocessing logic in production with the saved model

__Teal__ is in very early stage and a _lot_ of work is to be done. Please feel free to reach out if you'd like to help out!! :smile:

## Getting Started

Install would be using `pip`:

`pip install --user git+https://github.com/am1tyadav/teal.git`

### Preprocessing Model - Log Mel Spectrogram

```python
import tensorflow as tf
import teal

NUM_SAMPLES = 44100
SAMPLE_RATE = 22050
N_FFT = 2048
HOP_LEN = 512
N_MELS = 64

log_mel_model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(NUM_SAMPLES, )),
    teal.feature.MelSpectrogram(SAMPLE_RATE, N_FFT, HOP_LEN, N_MELS),
    teal.feature.PowerToDb()
])

# Save it as a Keras model or TF saved model
log_mel_model.save("log_mel.h5")
```

### Audio Data Augmentation Model

```python
import tensorflow as tf
import teal

NUM_SAMPLES = 44100

audio_augmentation_model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(NUM_SAMPLES, )),
    teal.augment.InversePolarity(0.5),
    teal.augment.RandomNoise(0.2),
    teal.augment.RandomGain(0.5)
])
```

For a detailed example, please take a look at [this notebook](examples/Audio%20Classifier.ipynb)

## Layers

### Preprocessing layers

* STFT - Computes Short Time Fourier Transform
* Spectrogram - Computes power spectrum
* MelSpectrogram - Computes mel spectrogram
* PowerToDb - Scales the power spectrum to db range
* NormalizeAudio - Scale audio to a range of (-1, 1)
* NormalizeSpectrum - Scale spectrogram to a range of (-1, 1)

More layers WIP

### Data augmentation layers

* InversePolarity - Inverts polarity of input audio
* RandomGain - Apply different random gain to different examples in a batch
* RandomNoise - Apply random noise to audio samples
* (WIP) NoiseBank - Apply noise from user given set of audio files - The audio files must be in 16-bit WAV format

More layers WIP
