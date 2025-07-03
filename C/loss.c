#ifndef LOSS_H
#define LOSS_H

double mse(const double *y_true, const double *y_pred, const int size);
double *mse_deriv(const double *y_true, const double *y_pred, 
                double *grad, const int size);
                
#endif