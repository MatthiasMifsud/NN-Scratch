import numpy as np

# Foward pass through the network
def predict(input, network):
    output = input
    for layer in network:
        output = layer.forward_pass(output)
    
    return output


def train(x_train, y_train, network, cost, cost_deriv, 
          epochs = 100, learning_rate = 0.01, verbose = True):
    
    error_history = []

    for times in range(epochs):
        error = 0

        for x, y in zip(x_train, y_train):
            #getting prediction                
            output = predict(x, network)

            # Cumalative error for current run

            error += cost(y, output)

            #Compuying cost w.r.t the output (Ready for back propogation)
            dCdY = cost_deriv(y, output)

            #Learning stage through gradient descent
            for layer in reversed(network):
                dCdY = layer.backward_pass(dCdY, learning_rate)

        error /= len(x)

        if verbose:
            print(f"Epoch: {times+1}/{epochs} | Error: {error}")



def test(x_test, y_test, network, verbose=True):
    correct_pred_count = 0
    for x, y in zip(x_test, y_test):
        output = predict(x, network)

        prediction = np.argmax(output)
        actual = np.argmax(y)

        if prediction == actual:
            correct_pred_count += 1

        if verbose == True:
            print(f"Prediction:{prediction} | Actual Value:{actual}")
    
    accuracy = 100 * correct_pred_count / len(y_test)
    print(f"Accuracy: {accuracy}%")


'''
        if len(error_history) <= 4:
                error_history.append(error)

        elif len(error_history) == 5:
            prev_history_mean = np.mean(error_history)
            
            if error >= prev_history_mean:
                learning_rate *= 8
                error_history.clear()
                print(f"learning rate changed to {learning_rate}")
'''