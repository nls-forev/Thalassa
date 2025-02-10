import numpy as np


class Activations:
    def __init__(self):
        self.gradients = {
            'relu': self.grad_relu,
            'lrelu': self.grad_lrelu,
            'sigmoid': self.grad_sigmoid,
            'tanh': self.grad_tanh,
            'softmax': self.grad_softmax
        }

    def relu(self, z: np.ndarray) -> np.ndarray:
        return np.maximum(0, z)

    def grad_relu(self, z: np.ndarray) -> np.ndarray:
        return (z > 0).astype(float)

    def lrelu(self, z: np.ndarray) -> np.ndarray:
        return np.where(z > 0, z, 0.01 * z)

    def grad_lrelu(self, z: np.ndarray) -> np.ndarray:
        return np.where(z > 0, 1.0, 0.01)

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))

    def grad_sigmoid(self, z: np.ndarray) -> np.ndarray:
        a = self.sigmoid(z)
        return a * (1 - a)

    def tanh(self, z: np.ndarray) -> np.ndarray:
        return np.tanh(z)

    def grad_tanh(self, z: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(z)**2

    def softmax(self, z: np.ndarray) -> np.ndarray:
        z_max = np.max(z, axis=0, keepdims=True)
        exp_z = np.exp(z - z_max)
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    def grad_softmax(self, z: np.ndarray) -> np.ndarray:
        pass  # Already handled in cross-entropy
