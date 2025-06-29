#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define INPUT_SIZE 28*28
#define OUTPUT_SIZE 10
#define HIDDEN_LAYER_SIZE 16


double weights1[HIDDEN_LAYER_SIZE][INPUT_SIZE];
double weights2[OUTPUT_SIZE][HIDDEN_LAYER_SIZE];

double bias1[HIDDEN_LAYER_SIZE];
double bias2[OUTPUT_SIZE];

double randn(){
    double u1 = (rand() + 1.0) / (RAND_MAX + 2.0);
    double u2 = (rand() + 1.0) / (RAND_MAX + 2.0);

    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

void create_matrix(double **matrix, const int row, const int column){

    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < column; j++)
        {
            matrix[i][j] = randn();
            printf("%f", matrix[i][j]);
        }
        printf("\n");
    }
}

int main(void){
    create_matrix(weights1, HIDDEN_LAYER_SIZE, INPUT_SIZE);
}

