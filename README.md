# teal

__teal__ (TensorFlow Audio Layers) contains a number of TensorFlow layers specifically written to be used with audio data

## Layers

### Base Layers

* AugmentationLayer - Any data augmentation layer is subclassed from this

### Feature layers

* STFT - Computes Short Time Fourier Transform
* Spectrogram - Computes power spectrum
* MelSpectrogram - Computes mel spectrogram. Mel filter bank is computed once when the layer is built
* PowerToDb - Scales the power spectrum to db . Useful to create log mel spectrogram if used on MelSpectrogram output

### Data augmentation layers

* InversePolarity - Inverts polarity of input audio
* RandomGain
* RandomNoise

### Other preprocessing layers

* Normalize
* NormalizeAudio
* NormalizeSpectrum

## Examples

__Create a Log Mel Spectrogram Model with tf.keras__

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
    teal.NormalizeAudio(),
    teal.MelSpectrogram(SAMPLE_RATE, N_FFT, HOP_LEN, N_MELS),
    teal.PowerToDb()
])
```

More examples coming soon
