#include "fullyconnected.h"

// goal is to get the output
void dense_forward(double *X, const int hidden_size, const int out_size, const int inp_size){
    double weighted_sum = 0.0;

    //hidden layer
    for (int i = 0; i < hidden_size; i++)
    {
        forward.Z1[i] = param.B_1[i];
        for (int j = 0; j < inp_size; j++)
        {
            forward.Z1[i] += X[j] * param.W_1[i * inp_size + j];

        }
    }

    //output layer
    for (int i = 0; i < out_size; i++)
    {
        forward.Z2[i] = param.B_2[i];
        for (int j = 0; j < hidden_size; j++)
        {
            forward.Z2[i] += forward.A1[j] * param.W_2[i * hidden_size + j];
        }
    }
}

void dense_backward(double *dCdY, double *dCdX, double *input, 
                    double *weights, double *biases, const int out_size, 
                    const int inp_size, double learning_rate){
    
    for (int i = 0; i < out_size; i++)
    {
        for (int j = 0; j < inp_size; j++)
        {
            double dCdW = dCdY[i] * input[j];
            weights[i * inp_size + j] -= learning_rate * dCdW;
        }
    }

    for (int i = 0; i < out_size; i++)
    {
        biases[i] -= learning_rate * dCdY[i];   
    }
}