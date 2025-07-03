#ifndef DATA_H
#define DATA_H

int file_size(FILE* file);
void fill_data(FILE* file, double *y, double *X, const int input_size);
void read_mnist(const char *train_path, const char *test_path,
                double **X_train, double **y_train, int *train_size,
                double **X_test, double **y_test, int *test_size);

double* one_hot_encode(const double *labels, int size, int num_classes);

#endif