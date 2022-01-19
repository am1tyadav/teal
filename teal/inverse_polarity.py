"""InversePolarity

Invert the polarity of input audio
"""

from teal.augment import AugmentationLayer


class InversePolarity(AugmentationLayer):
    """InversePolarity

    Invert the polarity of input audio
    """

    def __init__(self, chance: float, *args, **kwargs):
        super().__init__(chance=chance, *args, **kwargs)

    def compute_augmentation(self, inputs):
        return -1. * inputs
