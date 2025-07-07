#ifndef FULLYCONNECTED_H
#define FULLYCONNECTED_H

void create_randn_matrix(double *matrix, const int size);

void dense_forward(const double *X, const double *W, const double *B,
                    double *Z, const int input_size, const int out_size);
                    
void dense_backward(double *dCdY, double *dCdX, double *input, 
                    double *weights, double *biases, const int inp_size, 
                    const int out_size, double learning_rate);
#endif