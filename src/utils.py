from .losses import *
from .optimizers import *


def get_loss_function(self):
    losses = {
        'binary_crossentropy': BinaryCrossEntropy,
        'categorical_crossentropy': CategoricalCrossEntropy
    }
    return losses[self.loss](self.layers, self.cache, self.loss)


def get_optimizer_function(self, lr):
    optimizers = {
        'linear': Linear
    }
    return optimizers[self.optimizer](self.layers, lr)
