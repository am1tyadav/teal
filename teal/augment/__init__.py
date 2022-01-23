"""
Data augmentation layers

    InversePolarity
        Inverts polarity of input audio
    RandomGain
    RandomNoise
    NoiseBank

Base Layers:
    AugmentationLayer - Any data augmentation layer is subclassed from this
"""
from teal.augment.inverse_polarity import InversePolarity
from teal.augment.noise_bank import NoiseBank
from teal.augment.random_gain import RandomGain
from teal.augment.random_noise import RandomNoise
