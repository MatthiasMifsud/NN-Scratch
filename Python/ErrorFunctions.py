import numpy as np

#computing mean squared error and its derivative

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_derivative(y_true, y_pred):
    return  2 * (y_pred - y_true) / np.size(y_true)