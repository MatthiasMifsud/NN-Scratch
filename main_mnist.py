from FullyConnectedLayer import DenseLayer
from ActivationFunctions import ReLU, Softmax
from ErrorFunctions import mse, mse_derivative
from NetworkRunning import predict, train

from keras.datasets import mnist # type: ignore
from keras.utils import to_categorical # type: ignore

import numpy as np

def preprocessing(x, y):
    x = x.reshape(x.shape[0], 28*28, 1)
    x = x.astype("float32") / 255

    y = to_categorical(y)
    y = y.reshape(y.shape[0], 10, 1)

    return x, y

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train, Y_train = preprocessing(X_train, Y_train)
X_test, Y_test = preprocessing(X_test, Y_test)

INPUT_SIZE = 28*28
HIDDEN_SIZE = 16
OUTPUT_SIZE = 10

neural_network = [
    DenseLayer(INPUT_SIZE, HIDDEN_SIZE),
    ReLU(),
    DenseLayer(HIDDEN_SIZE, OUTPUT_SIZE),
    Softmax()
]

train(x_train=X_train, y_train=Y_train, network=neural_network,
      cost=mse, cost_deriv=mse_derivative, learning_rate=0.1, epochs=10)


count = 0
for x, y in zip(X_test, Y_test):
    output = predict(x, neural_network)

    prediction = np.argmax(output)
    actual = np.argmax(y)

    if prediction != actual:
        count += 1


    print('pred:', np.argmax(output), '\ttrue:', np.argmax(y))

print(100* (count/ len(Y_test)))