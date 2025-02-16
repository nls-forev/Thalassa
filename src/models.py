import pickle
from layers import *
from losses import *
from data import *
from utils import *
from preprocessing import *
from activation import Activations


class Model:
    def add(self):
        return NotImplementedError("Every subclass of Optimizer must implement this method!")

    def forward(self):
        return NotImplementedError("Every subclass of Optimizer must implement this method!")

    def compile(self):
        return NotImplementedError("Every subclass of Optimizer must implement this method!")

    def fit(self):
        return NotImplementedError("Every subclass of Optimizer must implement this method!")

    def predict(self):
        return NotImplementedError("Every subclass of Optimizer must implement this method!")


class Sequential(Model):
    def __init__(self):
        self.layers = []
        self.cache = {}
        self.optimizer = None
        self.loss = None
        self.training = True
        self.metrics = ['accuracy']

    # Add a layer to the layers stack
    def add(self, layer):
        if self.layers and isinstance(layer, Dense):
            prev_layer = None
            for l in reversed(self.layers):
                if not isinstance(l, Dropout):
                    prev_layer = l
                    break
            if prev_layer is not None:
                layer.input_size = prev_layer.units
        if isinstance(layer, Dense):
            layer.init_params()
        self.layers.append(layer)

    def forward(self, batch: np.ndarray) -> np.ndarray:
        a = batch.T
        dense_idx = 0  # Keep track of dense layer indices
        act = Activations()

        for i, layer in enumerate(self.layers):
            if isinstance(layer, Dropout):
                if self.training:  # Only apply dropout during training
                    # Apply dropout and store the mask in the cache for backpropagation.
                    a = layer.drop(a) / (1 - layer.drop_percent)
                    self.cache[f'dropout_mask{i}'] = layer.mask
                continue

            # For Dense layers
            z = np.dot(layer.w, a) + layer.b
            self.cache[f'z{dense_idx}'] = z
            self.cache[f'a_prev{dense_idx}'] = a

            a = act.activations[layer.activation](z)
            self.cache[f'a{dense_idx}'] = a
            dense_idx += 1
        return a.T

    def compile(self, optimizer=None, loss=None, metrics=['accuracy']):
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics

    def fit(self, x, y, epochs, lr=1e-3, batch_size=64, stop_at=None):
        y_encoded = OneHotEncoder(
            y) if self.loss == 'categorical_crossentropy' else y

        self.loss: Loss = get_loss_function(self)
        self.optimizer: Optimizer = get_optimizer_function(self, lr)

        for epoch in range(epochs):
            if stop_at and epoch == stop_at:
                break

            epoch_loss, correct, total = 0, 0, 0

            for i in range(0, len(x), batch_size):
                x_batch = x[i:i+batch_size]
                y_batch = y_encoded[i:i+batch_size]

                # Forward pass
                preds = self.forward(x_batch)

                # Compute loss
                epoch_loss += self.loss.compute_loss(y_batch,
                                                     preds) * len(x_batch)

                # Calculate gradients
                grads = self.loss.compute_grad(x_batch, y_batch)

                # update weights (optimizer)
                self.optimizer.update_weights(grads)

                # compute metrics
                correct += self.loss.statistics(y_batch, preds)
                total += len(y_batch)

            accuracy = correct / total
            print(
                f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(x):.4f}, Accuracy: {accuracy:.4f}")

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray):
        if isinstance(self.loss, str):
            pass

    def predict(self, x_test, y_test=None):
        y_hat = self.forward(x_test)
        return y_hat

    def reset(self):
        """
        Resets the model by clearing cache and reinitializing all trainable parameters.
        Use this to restart training with the same architecture but fresh weights.
        """
        self.cache = {}
        for layer in self.layers:
            if isinstance(layer, Dense):
                layer.init_params()


class ModelWrapper(Model):
    pass
