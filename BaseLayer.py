'''
constructing the base structure of each layer as the super class which will 
be inhereted by the fully connected and activation layers to perform forward
and backward propogation 
'''

class LayerStructure:

    def __init__(self):
        self.input = None
        self.output = None
    
    '''
    forward pass takes in the input neurons for the node we are evaluating and 
    returns the output neurons
    '''
    def forward_pass(self, input):
        pass

    '''
    takes in dC/dY (derivative of the error w.r.t the output) and updates the
    parameters and returns dC/dX (derivative of error w.r.t the input)
    '''
    def backward_pass(self, output_gradient, learning_rate):
        pass