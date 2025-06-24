def predict(input, network):
    output = input
    for layer in network:
        output = layer.forward_pass(output)
    
    return output


def train(x_train, y_train, network, cost, cost_deriv, 
          epochs = 100, learning_rate = 0.01, verbose = True):

    for times in range(epochs):
        error = 0

        for x, y in zip(x_train, y_train):
                
            output = predict(x, network)

            error += cost(y, output)

            dCdY = cost_deriv(y, output)

            #learning

            for layer in reversed(network):
                dCdY = layer.backward_pass(dCdY, learning_rate)

        error /= len(x)

        if verbose:
            print(f"epoch: {times+1}/{epochs} | error: {error}")