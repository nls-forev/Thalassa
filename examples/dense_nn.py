import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from thalassa.src.layers import *
from thalassa.src.models import Sequential
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train.reshape(-1, 784) / \
    255.0, x_test.reshape(-1, 784) / 255.0
y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)

# Create masks for digits 0 and 1 (flatten y to make masking easier)
mask_train = ((y_train.flatten() == 0) | (y_train.flatten() == 1))
mask_test = ((y_test.flatten() == 0) | (y_test.flatten() == 1))

# Apply masks to filter the data
x_train, y_train = x_train[mask_train], y_train[mask_train]
x_test, y_test = x_test[mask_test], y_test[mask_test]

model = Sequential()
model.add(Dense(128, input_size=784))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(10))
model.add(Dense(2, 'softmax'))
model.compile(optimizer='linear', loss='categorical_crossentropy')
model.fit(x_train, y_train, 5, batch_size=128, lr=1e-2)

print(np.argmax(model.predict(x_test, y_test), axis=1))
# y_hat, accuracy = model.predict(x_test[1].reshape(-1, 784), y_test[1])
# print(y_hat)
# print(np.argmax(y_hat, axis=1), accuracy)
