#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define INPUT_SIZE 28*28
#define OUTPUT_SIZE 10
#define HIDDEN_LAYER_SIZE 16

#define TEST_PATH "./data/mnist_test.csv"
#define TRAIN_PATH "./data/mnist_train.csv"
#define MAX_LINE_LENGTH 10000

struct Parameters{
    double *W_1;
    double *B_1;
    double *W_2;
    double *B_2;
} param;

struct Data{
    double *X_train;
    double *X_test;
    double *y_train;
    double *y_test;
} data;

struct Forward{
    double *A1;
    double *Z1;
    double *A2;
    double *Z2;
} forward;


void init_param(){
    param.W_1 = malloc(HIDDEN_LAYER_SIZE * INPUT_SIZE * sizeof(double));
    param.B_1 = malloc(HIDDEN_LAYER_SIZE * sizeof(double));
    param.W_2 = malloc(OUTPUT_SIZE * HIDDEN_LAYER_SIZE * sizeof(double));
    param.B_2 = malloc(OUTPUT_SIZE * sizeof(double));

    for (int i = 0; i < HIDDEN_LAYER_SIZE * INPUT_SIZE; i++)
    {
        param.W_1[i] = randn();
    }
    
    for (int i = 0; i < HIDDEN_LAYER_SIZE; i++)
    {
        param.B_1[i] = randn();
    }

    for (int i = 0; i < OUTPUT_SIZE * HIDDEN_LAYER_SIZE; i++)
    {
        param.W_2[i] = randn();
    }

    for (int i = 0; i < OUTPUT_SIZE; i++)
    {
        param.B_2[i] = randn();
    }
}
 
int main(void){
    read_data();
    init_param();

    free(data.y_train);
    free(data.y_test);
    free(data.X_train);
    free(data.X_test);
    return 0;
}
