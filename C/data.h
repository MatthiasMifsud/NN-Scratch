#ifndef DATA_H
#define DATA_H

#define MAX_LINE_LENGTH 10000
#define TEST_PATH "./data/mnist_test.csv"
#define TRAIN_PATH "./data/mnist_train.csv"

typedef struct{
    double *X_train;
    double *X_test;
    double *y_train;
    double *y_test;
    double *y_train_one_hot;
    double *y_test_one_hot;
}Data;

extern Data data;
extern int test_size;
extern int train_size;

void load_data(const int input_size, const int output_size);

void check_data(const int input_size, const int output_size);

void free_data();

#endif