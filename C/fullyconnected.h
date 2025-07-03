#ifndef FULLYCONNECTED_H
#define FULLYCONNECTED_H

double randn();

void dense_forward(const double *X, const double *W, const double *B,
                    double *Z, const int input_size, const int out_size);
                    
void dense_backward(double *dCdY, double *dCdX, double *input, 
                    double *weights, double *biases, const int out_size, 
                    const int inp_size, double learning_rate);
#endif