#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define INPUT_SIZE 28*28
#define OUTPUT_SIZE 10
#define HIDDEN_LAYER_SIZE 16

struct Parameters{
    double *weights_1;
    double *bias_1;
    double *weights_2;
    double *bias_2;
} param;

double randn(){
    double u1 = (rand() + 1.0) / (RAND_MAX + 2.0);
    double u2 = (rand() + 1.0) / (RAND_MAX + 2.0);

    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

void init_param(){
    param.weights_1 = malloc(HIDDEN_LAYER_SIZE * INPUT_SIZE * sizeof(double));
    param.bias_1 = malloc(HIDDEN_LAYER_SIZE * sizeof(double));
    param.weights_2 = malloc(OUTPUT_SIZE * HIDDEN_LAYER_SIZE * sizeof(double));
    param.bias_2 = malloc(OUTPUT_SIZE * sizeof(double));

    for (int i = 0; i < HIDDEN_LAYER_SIZE * INPUT_SIZE; i++)
    {
        param.weights_1[i] = randn();

        if (i <= HIDDEN_LAYER_SIZE) 
            param.bias_1[i] = randn();
    }

    for (int i = 0; i < OUTPUT_SIZE * HIDDEN_LAYER_SIZE; i++)
    {
        param.weights_2[i] = randn();

        if (i <= OUTPUT_SIZE)
            param.bias_2[i] = randn();
    }
}

double ReLU(double x){
    return (x > 0.0) ? x : 0.0;
}

double maximum(double *array, const int size){
    double max = array[0];

    for (int i = 1; i < size; i++)
    {
        if (array[i] > max)
            max = array[i];
    }
    return max;
}

double *Softmax(double *input, const int size){
    double max_input = maximum(input, size);
    double exponent[size];
    double exponent_sum = 0.0;

    for (int i = 0; i < size; i++)
    {
        exponent[i] = exp(input[i] - max_input);
        exponent_sum += exponent[i];
    }

    for (int i = 0; i < size; i++)
    {
        input[i] = exponent[i] / exponent_sum;
    }

    return input;
}

double *forward_prop(double input[INPUT_SIZE]){
    double weighted_sum = 0.0;
    
    // first part
    double *activation_1 = malloc(HIDDEN_LAYER_SIZE * sizeof(double));
    double *activation_2 = malloc(OUTPUT_SIZE * sizeof(double));

    for (int i = 0; i < HIDDEN_LAYER_SIZE; i++)
    {
        double sum = param.bias_1[i];
        for (int j = 0; j < INPUT_SIZE; j++)
        {
            sum += input[j] * param.weights_1[i * INPUT_SIZE + j];

        }
        activation_1[i] = ReLU(sum);
    }


    for (int i = 0; i < OUTPUT_SIZE; i++)
    {
        double sum = param.bias_1[i];
        for (int j = 0; j < HIDDEN_LAYER_SIZE; j++)
        {
            sum += input[j] * param.weights_2[i * HIDDEN_LAYER_SIZE + j];
        }
        activation_2[i] = sum;
    }

    Softmax(activation_2, OUTPUT_SIZE);

    free(activation_1);

    return activation_2;
}

double **backward_prop(){

}



int main(void){
    init_param();


    return 0;
}
