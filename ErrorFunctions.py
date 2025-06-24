import numpy as np

#computing mean squared error and its derivative

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_derivative(y_true, y_pred):
    return  2 * (y_pred - y_true) / np.size(y_true)
    
def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1-epsilon)
    return -np.sum(y_true * np.log(y_pred))

def cross_entropy_loss_deriv(y_pred, y_true):
    return y_pred - y_true