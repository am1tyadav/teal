# Teal - Audio Processing Layers for TensorFlow

Create TensorFlow layers and models for audio preprocessing and audio data augmentation:

:heavy_check_mark: No dependency other than TensorFlow

:heavy_check_mark: Can utilize GPU

:heavy_check_mark: Online preprocessing and data augmentation

:heavy_check_mark: Deploy preprocessing logic in production with the saved model

__Teal__ is in very early stage and a _lot_ of work is to be done. Please feel free to reach out if you'd like to help out!! :smile:

## Getting Started

Install would be using `pip`:

`pip install --user git+https://github.com/am1tyadav/teal.git`

### Example: Log Mel Spectrogram Model

```python
import tensorflow as tf
import teal

NUM_SAMPLES = 44100
SAMPLE_RATE = 22050
N_FFT = 2048
HOP_LEN = 512
N_MELS = 64

log_mel_model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(NUM_SAMPLES,)),
    teal.AudioToMelSpectrogram(SAMPLE_RATE, N_FFT, HOP_LEN, N_MELS),
    teal.PowerToDb()
])

# Save it as a Keras model or TF saved model
log_mel_model.save("log_mel.h5")
```

### Example: Audio Data Augmentation Model

```python
import teal.augment
import tensorflow as tf
import teal

NUM_SAMPLES = 44100

audio_augmentation_model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(NUM_SAMPLES,)),
    teal.InversePolarity(0.5),
    teal.RandomNoise(0.2),
    teal.RandomGain(0.5)
])
```

### Example: Audio Classification with TensorFlow and Teal

[Audio Classification with TensorFlow and Teal](examples/Audio_Classification_with_TensorFlow_and_Teal.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github.com/am1tyadav/teal/blob/main/examples/Audio_Classification_with_TensorFlow_and_Teal.ipynb)

## Layers

### Transformation Layers

* AudioToSTFT
* AudioToSpectrogram
* AudioToMelSpectrogram
* STFTToSpecAndPhase
* STFTToSpectrogram
* STFTToPhase
* SpectrogramToMelSpec
* PowerToDb
* DbToPower
* NormalizeAudio
* NormalizeSpectrum

### Data Augmentation Layers

* InversePolarity
* NoiseBank
* RandomGain
* RandomNoise
* PitchShift
