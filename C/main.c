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

double randn(){
    double u1 = (rand() + 1.0) / (RAND_MAX + 2.0);
    double u2 = (rand() + 1.0) / (RAND_MAX + 2.0);

    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

void init_param(){
    param.W_1 = malloc(HIDDEN_LAYER_SIZE * INPUT_SIZE * sizeof(double));
    param.B_1 = malloc(HIDDEN_LAYER_SIZE * sizeof(double));
    param.W_2 = malloc(OUTPUT_SIZE * HIDDEN_LAYER_SIZE * sizeof(double));
    param.B_2 = malloc(OUTPUT_SIZE * sizeof(double));

    for (int i = 0; i < HIDDEN_LAYER_SIZE * INPUT_SIZE; i++)
    {
        param.W_1[i] = randn();

        if (i <= HIDDEN_LAYER_SIZE) 
            param.B_1[i] = randn();
    }

    for (int i = 0; i < OUTPUT_SIZE * HIDDEN_LAYER_SIZE; i++)
    {
        param.W_2[i] = randn();

        if (i <= OUTPUT_SIZE)
            param.B_2[i] = randn();
    }
}

// activation functions 

double ReLU(double x){
    return (x > 0.0) ? x : 0.0;
}

// function to find maximum of an array
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

double ReLU_deriv(double x){

}

double Softmax_deriv(double x){

}

// aim is to get the output Y from an input X
double *forward_prop(double X[INPUT_SIZE]){
    double weighted_sum = 0.0;
    
    // first part
    double *activation_1 = malloc(HIDDEN_LAYER_SIZE * sizeof(double));
    double *activation_2 = malloc(OUTPUT_SIZE * sizeof(double));

    for (int i = 0; i < HIDDEN_LAYER_SIZE; i++)
    {
        double sum = param.B_1[i];
        for (int j = 0; j < INPUT_SIZE; j++)
        {
            sum += X[j] * param.W_1[i * INPUT_SIZE + j];

        }
        activation_1[i] = ReLU(sum);
    }


    for (int i = 0; i < OUTPUT_SIZE; i++)
    {
        double sum = param.B_2[i];
        for (int j = 0; j < HIDDEN_LAYER_SIZE; j++)
        {
            sum += X[j] * param.W_2[i * HIDDEN_LAYER_SIZE + j];
        }
        activation_2[i] = sum;
    }

    Softmax(activation_2, OUTPUT_SIZE);

    free(activation_1);

    return activation_2;
}

double *backward_prop(double Y[OUTPUT_SIZE]){
    
}

int file_size(FILE* file){
    int count = 0;
    char c;

    //reading one line at a time
    while ((c = fgetc(file)) != EOF){
        if (c == '\n') count++;
    }

    return (count > 0) ? count - 1 : 0; //decrementing count by 1 to skip header
}

void fill_data(FILE* file, int *y, double *X){
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

        for (int i = 0; i < INPUT_SIZE; i++)
        {
            token = strtok(NULL, ",");

            if (token != NULL)
                X[pos * INPUT_SIZE + i] = atof(token);
            else
                perror("ERROR: Incomplete pixel data");
        }
        pos++;
    }
}

void read_data(){
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

    int *y_train = malloc(train_size * sizeof(int));
    int *y_test = malloc(test_size * sizeof(int));
    double *X_train = malloc(INPUT_SIZE * train_size * sizeof(double));
    double *X_test = malloc(INPUT_SIZE * test_size * sizeof(double));

    fill_data(test_data, y_test, X_test);
    fill_data(train_data, y_train, X_train);

    fclose(test_data);
    fclose(train_data);

    free(y_train);
    free(y_test);
    free(X_train);
    free(X_test);
}


int main(void){
    read_data();
    return 0;
}
