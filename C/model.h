#ifndef MODEL_H
#define MODEL_H

#define TEST_PATH "./data/mnist_test.csv"
#define TRAIN_PATH "./data/mnist_train.csv"
#define INPUT_SIZE 28*28
#define OUTPUT_SIZE 10
#define HIDDEN_LAYER_SIZE 16

typedef struct{
    double *W1, *B1;
    double *W2, *B2;
} Model;

void init_param(Model *model);

void train_model(Model *model, double *X_train, double *y_train, 
             const int train_size, const int epochs, const double learning_rate);

void free_model(Model *model);

#endif