import numpy as np
from layers import *
from activation import *

class BinaryCrossEntropy:
    def __init__(self, cache: dict):
        self.cache = cache

    def compute_bce(self, y_true, y_pred):
        eps = 1e-8
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def backward_bce(self, x: np.ndarray, y: np.ndarray, layers, lr=1e-3):
        activations = Activations()
        batch_size = x.shape[0]
        output = self.cache[f'a{len(layers)-1}']

        # Compute the derivative for BCE
        dz = (output - y.T) / batch_size

        for i in reversed(range(len(layers))):
            layer = layers[i]

            if isinstance(layer, Dropout):
                continue  # Skip dropout layers

            # Find previous activation
            a_prev_idx = i - 1
            while a_prev_idx >= 0 and isinstance(layers[a_prev_idx], Dropout):
                a_prev_idx -= 1
            a_prev = self.cache[f'a{a_prev_idx}'] if a_prev_idx >= 0 else x.T

            # Compute gradients
            dw = np.dot(dz, a_prev.T) / batch_size
            db = np.mean(dz, axis=1, keepdims=True)

            # Update parameters
            layer.w -= lr * dw
            layer.b -= lr * db

            # Compute gradient for next layer
            if i > 0:
                z_prev_idx = i - 1
                while z_prev_idx >= 0 and isinstance(layers[z_prev_idx], Dropout):
                    z_prev_idx -= 1

                if z_prev_idx >= 0:
                    z_prev = self.cache[f'z{z_prev_idx}']
                    # Example: 'relu', 'sigmoid'
                    activation_name = layers[z_prev_idx].activation

                    # Efficient gradient lookup
                    grad_function = activations.gradients.get(activation_name)

                    if grad_function:
                        dz = np.dot(layer.w.T, dz) * grad_function(z_prev)
                    else:
                        raise ValueError(
                            f"Gradient function for '{activation_name}' not found!")

    def statistics(self, y: np.ndarray, y_pred: np.ndarray, metrics=['accuracy']):
        if 'accuracy' in metrics:
            binary_preds = (y_pred > 0.5).astype(int)
            return np.sum(binary_preds == y)


class CategoricalCrossEntropy:
    def __init__(self, cache: dict):
        self.cache = cache

    def compute_ce(self, y_true, y_pred):
        eps = 1e-8
        y_pred = np.clip(y_pred, eps, 1 - eps)
        # Sum over classes and average over batch
        # Fixed reduction
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    def backward_ce(self, x: np.ndarray, y: np.ndarray, layers: list, lr: float):
        activations = Activations()
        batch_size = x.shape[0]

        # Find last dense layer index
        last_dense_idx = sum(
            1 for layer in layers if isinstance(layer, Dense)) - 1

        # Get output layer activation
        a_last = self.cache[f'a{last_dense_idx}']

        # Compute the derivative for softmax + cross-entropy
        dz = a_last - y.T
        dense_idx = last_dense_idx

        for i in reversed(range(len(layers))):
            layer = layers[i]

            if isinstance(layer, Dropout):
                dz = dz * \
                    self.cache[f'dropout_mask{i}'] / (1 - layer.drop_percent)
                continue

            if dense_idx == 0:
                a_prev = x.T
            else:
                a_prev = self.cache[f'a_prev{dense_idx}']

            # Compute gradients
            dw = np.dot(dz, a_prev.T) / batch_size
            db = np.sum(dz, axis=1, keepdims=True) / batch_size

            # Update parameters
            layer.w -= lr * dw
            layer.b -= lr * db

            # Compute gradient for next layer
            if dense_idx > 0:
                z_prev = self.cache[f'z{dense_idx-1}']
                # Example: 'relu', 'sigmoid'
                activation_name = layers[i - 1].activation

                # Efficient lookup for the activation gradient function
                grad_function = activations.gradients.get(activation_name)

                if grad_function:
                    dz = np.dot(layer.w.T, dz) * grad_function(z_prev)
                else:
                    raise ValueError(
                        f"Gradient function for '{activation_name}' not found!")

            dense_idx -= 1

    def statistics(self, y: np.ndarray, y_pred: np.ndarray, metrics=['accuracy']):
        if 'accuracy' in metrics:
            pred_classes = np.argmax(y_pred, axis=1)
            true_classes = np.argmax(y, axis=1)
            return np.sum(pred_classes == true_classes)
