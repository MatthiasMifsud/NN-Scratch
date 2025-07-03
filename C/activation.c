#include "activation.h"

// activation functions 
double ReLU(double x){
    return (x > 0.0) ? x : 0.0;
}

double ReLU_deriv(double x){
    return (x > 0) ? 1.0 : 0.0;
}

