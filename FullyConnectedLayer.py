from BaseLayer import LayerStructure

import numpy as np

#connecting between layers
#always inheriting the same Layer structure
class DenseLayer(LayerStructure):

    def __init__(self, input_size, output_size):
        
        #initilising random weights and biases with correct dimentions
    
        self.weights = np.random.randn(output_size, input_size) # j x i
        self.bias = np.random.randn(output_size, 1) # j x 1

    #goal is to output Y = W . X + B
    def forward_pass(self, input_neurons):
        self.input = input_neurons
        return np.dot(self.weights, self.input) + self.bias #resultant dimention: j x 1
    '''
    taking input the derivative of the error w.r.t
    to the output(calculated in foward pass)
    
    goal is to return the derivative of the error w.r.t the input and perform
    gradient descent  
    '''
    def backward_pass(self, dCdY, learning_rate):
        #derivative of error w.r.t the weights
        dCdW = np.dot(dCdY, self.input.T)

        #precomputing the derivative of cost w.r.t the input
        dCdX = np.dot(self.weights.T, dCdY)

        #updating parameters using gradient descent

        self.weights -= dCdW * learning_rate 

        #using dCdY since from chain rule dCdB == dCdY
        self.bias -= dCdY * learning_rate

        return dCdX

