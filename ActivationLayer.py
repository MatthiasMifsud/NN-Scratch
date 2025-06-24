from BaseLayer import LayerStructure

import numpy as np

'''
here we take some input neurons and pass them through an activation function 
so the input and output layers have the same shape
equation used y = f(x) where f() is the activation function used
'''

class Activation(LayerStructure):

    def __init__(self, activation_func, activation_func_deriv):
        self.activation_function = activation_func
        self.activation_function_deriv = activation_func_deriv

    #passing activation to the input
    def forward_pass(self, input):
        #input is the output of the foward pass from the fully connected layer
        self.input = input
        return self.activation_function(self.input)
    
    '''
    goal here is to return the cost function w.r.t the input from given dCdY
    which is the element wise operation of the dCdY and the derivative
    of the activation function (found using chain rule )
    '''
    def backward_pass(self, dCdY, learning_rate):
        return np.multiply(dCdY, self.activation_function_deriv(self.input))