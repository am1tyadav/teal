from teal.augment import AugmentationLayer


class InversePolarity(AugmentationLayer):
    def __init__(self, chance: float):
        super(InversePolarity, self).__init__(chance=chance)

    def compute_augmentation(self, inputs):
        return -1. * inputs
