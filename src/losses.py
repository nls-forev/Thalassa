import numpy as np
from .layers import *
from .activation import *


class Loss:
    def compute_loss(self):
        return NotImplementedError("Every subclass of Optimizer must implement this method!")

    def compute_grad(self):
        return NotImplementedError("Every subclass of Optimizer must implement this method!")


class CrossEntropyLoss(Loss):
    def __init__(self, layers, cache: dict, name: str):
        self.layers = layers
        self.cache = cache
        self.name = name
        self.grads = {}

    def compute_grad_common(self, x: np.ndarray, y: np.ndarray):
        activations = Activations()
        batch_size = len(x)
        # Find last Dense layer index
        last_dense_idx = sum(
            1 for layer in self.layers if isinstance(layer, Dense)) - 1
        # Get output layer activation
        a_last = self.cache[f'a{last_dense_idx}']
        # Common derivative: (y_pred - y) / batch_size
        dz = (a_last - y.T) / batch_size
        dense_idx = last_dense_idx

        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]

            if isinstance(layer, Dropout):
                dz = dz * \
                    self.cache[f'dropout_mask{i}'] / (1 - layer.drop_percent)
                continue

            if dense_idx == 0:
                a_prev = x.T
            else:
                a_prev = self.cache[f'a_prev{dense_idx}']

            # Compute gradients for weights and biases
            dw = np.dot(dz, a_prev.T)
            db = np.sum(dz, axis=1, keepdims=True)
            self.grads[i] = {'w': dw, 'b': db}

            # Propagate gradient to previous Dense layer
            if dense_idx > 0:
                z_prev = self.cache[f'z{dense_idx-1}']
                prev_layer_idx = i - 1
                while prev_layer_idx >= 0 and not isinstance(self.layers[prev_layer_idx], Dense):
                    prev_layer_idx -= 1
                if prev_layer_idx >= 0:
                    activation_name = self.layers[prev_layer_idx].activation
                    grad_function = activations.gradients.get(activation_name)
                    if grad_function:
                        dz = np.dot(layer.w.T, dz) * grad_function(z_prev)
            dense_idx -= 1

        return self.grads


class BinaryCrossEntropy(CrossEntropyLoss):

    def compute_loss(self, y_true, y_pred):
        eps = 1e-8
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def compute_grad(self, x: np.ndarray, y: np.ndarray):
        return self.compute_grad_common(x, y)

    def statistics(self, y: np.ndarray, y_pred: np.ndarray, metrics=['accuracy']):
        if 'accuracy' in metrics:
            binary_preds = (y_pred > 0.5).astype(int)
            return np.sum(binary_preds == y)


class CategoricalCrossEntropy(CrossEntropyLoss):
    def compute_loss(self, y_true, y_pred):
        eps = 1e-8
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    def compute_grad(self, x: np.ndarray, y: np.ndarray):
        return self.compute_grad_common(x, y)

    def statistics(self, y: np.ndarray, y_pred: np.ndarray, metrics=['accuracy']):
        if 'accuracy' in metrics:
            pred_classes = np.argmax(y_pred, axis=1)
            true_classes = np.argmax(y, axis=1)
            return np.sum(pred_classes == true_classes)
