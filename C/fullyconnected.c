#include <stdlib.h>
#include <math.h>
#include "fullyconnected.h"

// random fucntion for random weights and biases
double randn(){
    double u1 = (rand() + 1.0) / (RAND_MAX + 2.0);
    double u2 = (rand() + 1.0) / (RAND_MAX + 2.0);

    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

// goal is to get the output
void dense_forward(const double *X, const double *W, const double *B,
                    double *Z, const int input_size, const int out_size){

    for (int i = 0; i < out_size; i++)
    {
        Z[i] = B[i];
        for (int j = 0; j < input_size; j++)
        {
            Z[i] += X[j] * W[i * input_size + j];
        }
    }
}

// goal is to compute the deriv of the cost w.r.t to the input 
// and computing gradient descent
void dense_backward(double *dCdY, double *dCdX, double *input, 
                    double *W, double *B, const int out_size, 
                    const int input_size, double learning_rate){
    
    
    //computing the derivative of cost w.r.t the weights and performing 
    //gradient descent on the weights
    for (int i = 0; i < out_size; i++)
    {
        for (int j = 0; j < input_size; j++)
        {
            double dCdW = dCdY[i] * input[j];
            W[i * input_size + j] -= learning_rate * dCdW;
        }
    }

    //performing gradient descent on the bias 
    for (int i = 0; i < out_size; i++)
    {
        B[i] -= learning_rate * dCdY[i];   
    }
}