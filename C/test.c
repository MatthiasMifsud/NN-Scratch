#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "data.h"

#define MAX_LINE_LENGTH 10000

int file_size(FILE* file){
    int count = 0;
    char c;

    while ((c = fgetc(file)) != EOF){
        if (c == '\n') count++;
    }
    return (count > 0) ? count - 1 : 0; // skip header
}

void fill_data(FILE* file, double *y, double *X, int input_size){
    char line[MAX_LINE_LENGTH];
    int pos = 0;

    fgets(line, sizeof(line), file); // skip header

    while (fgets(line, sizeof(line), file)){
        line[strcspn(line, "\r\n")] = '\0';

        char *token = strtok(line, ",");
        y[pos] = atoi(token);

        for (int i = 0; i < input_size; i++){
            token = strtok(NULL, ",");
            if (token != NULL)
                X[pos * input_size + i] = atof(token) / 255.0;
            else
                perror("ERROR: Incomplete pixel data");
        }
        pos++;
    }
}

void read_mnist(const char *train_path, const char *test_path,
                double **X_train, double **y_train, int *train_size,
                double **X_test, double **y_test, int *test_size){

    FILE* train_file = fopen(train_path, "r");
    FILE* test_file = fopen(test_path, "r");

    if (!train_file || !test_file) {
        perror("ERROR: Could not open train or test file");
        exit(1);
    }

    *train_size = file_size(train_file);
    rewind(train_file);
    *test_size = file_size(test_file);
    rewind(test_file);

    *y_train = malloc(*train_size * sizeof(double));
    *y_test = malloc(*test_size * sizeof(double));
    *X_train = malloc(INPUT_SIZE * (*train_size) * sizeof(double));
    *X_test = malloc(INPUT_SIZE * (*test_size) * sizeof(double));

    fill_data(train_file, *y_train, *X_train, INPUT_SIZE);
    fill_data(test_file, *y_test, *X_test, INPUT_SIZE);

    fclose(train_file);
    fclose(test_file);
}

double* one_hot_encode(const double *labels, int size, int num_classes){
    double *one_hot = malloc(size * num_classes * sizeof(double));
    for (int i = 0; i < size; i++){
        for (int j = 0; j < num_classes; j++){
            one_hot[i * num_classes + j] = (labels[i] == j) ? 1.0 : 0.0;
        }
    }
    return one_hot;
}
