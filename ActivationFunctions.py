from ActivationLayer import Activation

import numpy as np

'''
A library full of activation functions which are inheriting from the 
Activation Layer class so that we pass both the activation function
and its derivtive to the activation layer
'''

#ReLU 
class ReLU(Activation):
    def __init__(self):

        def ReLU(x):
            return np.maximum(0, x).astype(float)

        def ReLU_derivative(x):
            return (x > 0).astype(float) # returns 1 if we have positive value... 0 otherwise

        super().__init__(ReLU, ReLU_derivative)


'''
for Softmax each output variable is dependant on each of the input variable
this is shown by the summation sign in the denominator 

given this we cannon inheret the Activation layer but have to inheret the 
Base Layer instead as we need to consider this as a Fully Connected layer with
both foward and backward passes of its own 
'''

from BaseLayer import LayerStructure

#stable softmax
class Softmax(LayerStructure):
    def forward_pass(self, input):
        stable_input = input - np.max(input)
        exponent = np.exp(stable_input)
        self.output = exponent / np.sum(exponent)
        return self.output

    def backward_pass(self, output_gradient, learning_rate):
        s = self.output.reshape(-1, 1)
        jacobian = np.diagflat(s) - np.dot(s, s.T)

        return np.dot(jacobian, output_gradient)