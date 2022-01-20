# teal

__teal__ is a library of TensorFlow layers written for audio data preprocessing

Easily create TensorFlow models for audio preprocessing and audio data augmentation:

:heavy_check_mark: No dependency other than TensorFlow

:heavy_check_mark: Computations for preprocessing and data augmentation can utilize GPU

:heavy_check_mark: Online preprocessing and data augmentation

:heavy_check_mark: Deploy preprocessing logic in production by using the saved preprocessing model - i.e. no need to re-implement preprocessing logic in production

__teal__ is in very early stage and a _lot_ of work is to be done. Please feel free to reach out if you'd like to help out!! :smile:

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
    teal.MelSpectrogram(SAMPLE_RATE, N_FFT, HOP_LEN, N_MELS),
    teal.PowerToDb()
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
    teal.InversePolarity(0.5),
    teal.RandomNoise(0.2),
    teal.RandomGain(0.5)
])
```

## Layers

### Preprocessing layers

* STFT - Computes Short Time Fourier Transform
* Spectrogram - Computes power spectrum
* MelSpectrogram - Computes mel spectrogram. Mel filter bank is computed once when the layer is built
* PowerToDb - Scales the power spectrum to db . Useful to create log mel spectrogram if used on MelSpectrogram output
* NormalizeAudio - Scale audio to a range of (-1, 1)
* NormalizeSpectrum - Scale spectrogram to a range of (-1, 1)


### Data augmentation layers

* InversePolarity - Inverts polarity of input audio
* RandomGain - Apply different random gain to different examples in a batch
* RandomNoise - Apply random noise to audio samples
