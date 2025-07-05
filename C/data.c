#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "data.h"

//handeling the MNIST data 
void init_data(const int train_size, const int test_size, 
            const int input_size, const int output_size){

    data.X_test = malloc(test_size * input_size * sizeof(double));
    data.X_train = malloc(train_size * input_size * sizeof(double));
    data.y_test = malloc(test_size * sizeof(double));
    data.y_train = malloc(train_size * sizeof(double));
    data.y_test_one_hot = malloc(test_size * output_size * sizeof(double));
    data.y_train_one_hot = malloc(train_size * output_size * sizeof(double));
}

void free_data(){
    free(data.X_test);
    free(data.X_train);
    free(data.y_test_one_hot);
    free(data.y_train_one_hot);
}

void one_hot(double *y, double *y_one_hot, const int size, const int output_size){
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < output_size; j++)
        {
            y_one_hot[i * output_size + j] = (y[i] == j) ? 1.0 : 0.0;
        }   
    }
}

int file_size(FILE* file){
    int count = 0;
    char c;

    //reading one line at a time
    while ((c = fgetc(file)) != EOF){
        if (c == '\n') count++;
    }
    rewind(file);
    return (count > 0) ? count - 1 : 0; //decrementing count by 1 to skip header
}

void read_data(FILE* file, double *y, double *X, const int input_size){
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

void load_data(const int input_size, const int output_size){
    FILE* test_data = fopen(TEST_PATH, "r");
    FILE* train_data = fopen(TRAIN_PATH, "r");

    if (!test_data || !train_data){
        perror("ERROR: Cant find/open file!\n");
        exit(1);
    }

    int pos = 0;

    const int train_size = file_size(train_data);
    const int test_size = file_size(test_data);

    init_data(train_size, test_size, input_size, output_size);

    read_data(test_data, data.y_test, data.X_test, input_size);
    read_data(train_data, data.y_train, data.X_train, input_size);
    one_hot(data.y_test, data.y_test_one_hot, test_size, output_size);
    one_hot(data.y_train, data.y_train_one_hot, train_size, output_size);

    free(data.y_test);
    free(data.y_train);

    fclose(test_data);
    fclose(train_data);
}