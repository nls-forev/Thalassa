import numpy as np


class Optimizer:
    def update_weights(self):
        return NotImplementedError("Every subclass of Optimizer must implement this method!")


class Linear(Optimizer):
    def __init__(self, layers, lr):
        self.layers = layers
        self.lr = lr

    def update_weights(self, grads):
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'drop'):
                # No weights to update for dropout layers
                continue
            if i in grads:
                layer.w -= self.lr * grads[i]['w']
                layer.b -= self.lr * grads[i]['b']


class RMSProp(Optimizer):
    def __init__(self, layers, lr):
        self.layers = layers
        self.lr = lr

    def update_weights(self):
        pass
