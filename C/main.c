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
    int *y_train;
    int *y_test;
} data;

struct Forward{
    double *A1;
    double *Z1;
    double *A2;
    double *Z2;
} forward;

struct Backward{
    double *dZ1;
    double *dZ2;
    double *dW1;
    double *dW2;
    double *dB1;
    double *dB2;
} backward;

// random fucntion for random weights and biases
double randn(){
    double u1 = (rand() + 1.0) / (RAND_MAX + 2.0);
    double u2 = (rand() + 1.0) / (RAND_MAX + 2.0);

    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
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

// activation functions 
double ReLU(double x){
    return (x > 0.0) ? x : 0.0;
}

double ReLU_deriv(double x){
    return (x > 0) ? 1.0 : 0.0;
}

// loss function 

double mse(){
    
}

// goal is to get the output
void forward_prop(double X[INPUT_SIZE]){
    double weighted_sum = 0.0;
    
    // first part
    forward.Z1 = malloc(HIDDEN_LAYER_SIZE * sizeof(double));
    forward.A1 = malloc(HIDDEN_LAYER_SIZE * sizeof(double));
    forward.Z2 = malloc(OUTPUT_SIZE * sizeof(double));
    forward.A2 = malloc(OUTPUT_SIZE * sizeof(double));

    //hidden layer
    for (int i = 0; i < HIDDEN_LAYER_SIZE; i++)
    {
        forward.Z1[i] = param.B_1[i];
        for (int j = 0; j < INPUT_SIZE; j++)
        {
            forward.Z1[i] += X[j] * param.W_1[i * INPUT_SIZE + j];

        }
        forward.A1[i] = ReLU(forward.Z1[i]);
    }

    //output layer
    for (int i = 0; i < OUTPUT_SIZE; i++)
    {
        forward.Z2[i] = param.B_2[i];
        for (int j = 0; j < HIDDEN_LAYER_SIZE; j++)
        {
            forward.Z2[i] += forward.A1[j] * param.W_2[i * HIDDEN_LAYER_SIZE + j];
        }
        forward.A2[i] = forward.Z2[i];
    }

    Softmax(forward.A2, OUTPUT_SIZE);
}

void backward_prop(double Y[OUTPUT_SIZE], double learning_rate){

    //output layer

    backward.dZ2 = malloc(OUTPUT_SIZE * sizeof(double));
    for (int i = 0; i < OUTPUT_SIZE; i++)
    {
        backward.dZ2[i] = forward.A2[i] - Y[i];
    }
    
    for (int i = 0; i < OUTPUT_SIZE; i++)
    {
        for (int j = 0; j < HIDDEN_LAYER_SIZE; j++)
        {
            param.W_2[i * HIDDEN_LAYER_SIZE + j] -= learning_rate * backward.dZ2[i] * forward.A1[j];
        }   
        param.B_2[i] -= learning_rate * backward.dZ2[i];
    }

    //hidden layer
    backward.dZ1 = malloc(HIDDEN_LAYER_SIZE * sizeof(double));

    }
    




  
}

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
                X[pos * INPUT_SIZE + i] = atof(token) / 255.0;
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

    data.y_train = malloc(train_size * sizeof(int));
    data.y_test = malloc(test_size * sizeof(int));
    data.X_train = malloc(INPUT_SIZE * train_size * sizeof(double));
    data.X_test = malloc(INPUT_SIZE * test_size * sizeof(double));

    fill_data(test_data, data.y_test, data.X_test);
    fill_data(train_data, data.y_train, data.X_train);

    fclose(test_data);
    fclose(train_data);
}

int main(void){
    init_param();

    free(data.y_train);
    free(data.y_test);
    free(data.X_train);
    free(data.X_test);
    return 0;
}
