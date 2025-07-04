#ifndef DATA_H
#define DATA_H

#define TEST_PATH "./data/mnist_test.csv"
#define TRAIN_PATH "./data/mnist_train.csv"
#define MAX_LINE_LENGTH 10000

struct Data{
    double *X_train;
    double *X_test;
    double *y_train;
    double *y_test;
    double *y_train_one_hot;
    double *y_test_one_hot;
}data;

int file_size(FILE* file);
void fill_data(FILE* file, double *y, double *X, const int input_size);
void read_mnist(const char *train_path, const char *test_path,
                double **X_train, double **y_train, int *train_size,
                double **X_test, double **y_test, int *test_size);

double* one_hot_encode(const double *labels, int size, int num_classes);

#endif