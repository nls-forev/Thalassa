import numpy as np


class Layer:
    pass


class Dense(Layer):
    def __init__(self, units, activation='relu', input_size=None):  # Fixed input_size default
        self.units = units
        self.activation = activation
        self.input_size = input_size  # Now properly handles sequential connection
        self.w = None
        self.b = None

    def init_params(self):
        if self.input_size is None:
            raise ValueError("Must specify input_size for first layer")
        last_input_size = np.prod(self.input_size)

        # Initialize weights based on activation
        if self.activation in ['relu', 'lrelu', 'softmax']:  # Include softmax
            self.w = np.random.randn(
                self.units, last_input_size) * np.sqrt(2/last_input_size)
        elif self.activation in ['tanh', 'sigmoid']:
            self.w = np.random.randn(
                self.units, last_input_size) * np.sqrt(1/(last_input_size + self.units))
        else:
            self.w = np.random.randn(self.units, last_input_size) * 0.01
        self.b = np.zeros((self.units, 1))


class Dropout(Layer):
    def __init__(self, drop_percent):
        self.drop_percent = drop_percent

    def drop(self, a: np.ndarray):
        self.mask = np.random.binomial(
            1, 1-self.drop_percent, size=a.shape)
        return a * self.mask
