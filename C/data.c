#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "data.h"

//handeling the MNIST data 
int file_size(FILE* file){
    int count = 0;
    char c;

    //reading one line at a time
    while ((c = fgetc(file)) != EOF){
        if (c == '\n') count++;
    }
    return (count > 0) ? count - 1 : 0; //decrementing count by 1 to skip header
}

void fill_data(FILE* file, double *y, double *X, const int input_size){
    char line[MAX_LINE_LENGTH];
    int pos = 0;

    //reading the test data
    //skipping the first line (header)
    fgets(line, sizeof(line), file);

    //reading one line at a time
    while (fgets(line, sizeof(line), file)){

        line[strcspn(line, "\r\n")] = '\0';

        //tokenising each character in the line and separating labels and pixels
        char *token = strtok(line, ",");
        y[pos] = atoi(token); //pointer to the first token is the label

        for (int i = 0; i < input_size; i++)
        {
            token = strtok(NULL, ",");

            if (token != NULL)
                X[pos * input_size + i] = atof(token) / 255.0;
            else
                perror("ERROR: Incomplete pixel data");
        }
        pos++;
    }
}

double *one_hot(double *y, const int size, const int output_size){
    double *one_hot_y = malloc(size * output_size * sizeof(double));
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < output_size; j++)
        {
            one_hot_y[i * output_size + j] = (y[i] == j) ? 1.0 : 0.0;
        }   
    }
    return one_hot_y;
}

void read_data(const int input_size){
    FILE* test_data = fopen(TEST_PATH, "r");
    FILE* train_data = fopen(TRAIN_PATH, "r");

    if (!test_data || !train_data){
        perror("ERROR: Cant find/open file!\n");
        exit(1);
    }

    int pos = 0;
    const int train_size = file_size(train_data);
    rewind(train_data);
    const int test_size = file_size(test_data);
    rewind(test_data);

    data.y_train = malloc(train_size * sizeof(int));
    data.y_test = malloc(test_size * sizeof(int));
    data.X_train = malloc(input_size * train_size * sizeof(double));
    data.X_test = malloc(input_size * test_size * sizeof(double));

    fill_data(test_data, data.y_test, data.X_test);
    fill_data(train_data, data.y_train, data.X_train);


    fclose(test_data);
    fclose(train_data);
}
