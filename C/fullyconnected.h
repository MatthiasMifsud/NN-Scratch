#ifndef FULLYCONNECTED_H
#define FULLYCONNECTED_H

void dense_forward(double *X, const int hidden_size, const int out_size, const int inp_size);
void dense_backward(double *dCdY, double *dCdX, double *input, 
                    double *weights, double *biases, const int out_size, 
                    const int inp_size, double learning_rate);
#endif