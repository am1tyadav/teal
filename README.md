# teal
A set of audio specific layers for TensorFlow and Keras

## Feature Layers

```python
from teal.feature import STFT, LogMelSpectrogram
```

## Augmentation Layers

Augmentation layers are used only in the training phase and are bypassed 
during the evaluation and prediction automatically

```python
from teal.augment import RandomNoise, RandomGain, InversePolarity
```

## Usage

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
from teal.augment import RandomNoise, RandomGain, InversePolarity
from teal.feature import STFT, LogMelSpectrogram


model = Sequential([
    Input(shape=(num_samples, )),
    InversePolarity(0.5),
    RandomGain(0.5),
    RandomNoise(0.5),
    LogMelSpectrogram(sample_rate, n_fft, hop_length, n_mels)
])
```
