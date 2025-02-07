from layers import *
from losses import *
from activation import Activations


class Sequential:
    def __init__(self):
        self.layers = []
        self.cache = {}
        self.optimizer = 'rmsprop'
        self.loss = ''
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

        for i, layer in enumerate(self.layers):
            if isinstance(layer, Dropout):
                if self.training:  # Only apply dropout during training
                    # Scale by dropout rate
                    a = layer.drop(a) / (1-layer.drop_percent)
                continue

            # For Dense layers
            z = np.dot(layer.w, a) + layer.b
            self.cache[f'z{dense_idx}'] = z
            self.cache[f'a_prev{dense_idx}'] = a

            a = self.activate(z, layer.activation)
            self.cache[f'a{dense_idx}'] = a
            dense_idx += 1
        return a.T

    def compile(self, optimizer='rmsprop', loss='bce', metrics=['accuracy']):
        if loss == 'binary_crossentropy':
            self.loss = 'bce'
        elif loss == 'categorical_crossentropy':
            self.loss = 'ce'
        else:
            raise NotImplementedError("Unsupported loss function")
        self.optimizer = optimizer
        self.metrics = metrics

    def fit(self, x, y, epochs, lr=1e-3, batch_size=64, stop_at=None):
        num_classes = np.max(y) + 1
        y_encoded = np.eye(num_classes)[
            y.squeeze()] if self.loss == 'ce' else y

        for epoch in range(epochs):
            if stop_at and epoch == stop_at:
                break

            epoch_loss = 0
            correct = 0
            total = 0

            for i in range(0, len(x), batch_size):
                loss = 0.0
                x_batch = x[i:i+batch_size]
                y_batch = y_encoded[i:i+batch_size]

                # Forward pass
                preds = self.forward(x_batch)

                # Compute loss
                if self.loss == 'ce':
                    loss = CategoricalCrossEntropy(self.cache)
                    loss.compute_ce(y_batch, preds)
                    epoch_loss += loss.compute_ce(y_batch,
                                                  preds) * len(x_batch)
                elif self.loss == 'bce':
                    loss = BinaryCrossEntropy(self.cache)
                    epoch_loss += loss.compute_bce(y_batch,
                                                   preds) * len(x_batch)

                # Backward pass
                if self.loss == 'ce':
                    loss.backward_ce(x_batch, y_batch, self.layers, lr)
                elif self.loss == 'bce':
                    loss.backward_bce(x_batch, y_batch, self.layers, lr)

                # Calculate accuracy
                if self.loss == 'ce':
                    correct += loss.statistics(y_batch, preds)
                    total += len(y_batch)
                elif self.loss == 'bce':
                    correct += loss.statistics(y_batch, preds)
                    total += len(y_batch)

            accuracy = correct / total
            print(
                f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(x):.4f}, Accuracy: {accuracy:.4f}")

    def activate(self, z, activation):
        # Activation functions.
        act = Activations()
        if activation == 'relu':
            return act.relu(z)
        elif activation == 'sigmoid':
            return act.sigmoid(z)
        elif activation == 'softmax':
            return act.softmax(z)
        elif activation == 'lrelu':
            return act.lrelu(z)
        elif activation == 'tanh':
            return act.tanh(z)

    def predict(self, x_test, y_test=None):
        y_hat = self.forward(x_test)
        if y_test is not None:
            if self.loss == 'bce':
                binary_preds = (y_hat > 0.5).astype(int)
                accuracy = np.sum(binary_preds == y_test) / len(y_test) * 100
            elif self.loss == 'ce':
                pred_classes = np.argmax(y_hat, axis=1)
                true_classes = y_test.squeeze().astype(int)  # Directly use labels
                accuracy = np.sum(
                    pred_classes == true_classes) / len(y_test) * 100
            return y_hat, accuracy
        return y_hat

    def save(self, path: str):
        np.save(rf'{path}', self.cache, allow_pickle=True)
        print("File saved!")

    def reset(self):
        self.cache = {}
