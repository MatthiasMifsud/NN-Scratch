#include "fullyconnected.h"
#include "activation.h"
#include "data.h"
#include "loss.h"
#include "model.h"

#include <stdio.h>
#include <stdlib.h>

void init_param(Model *model){
    create_randn_matrix(model->W1, HIDDEN_LAYER_SIZE * INPUT_SIZE);
    create_randn_matrix(model->B1, HIDDEN_LAYER_SIZE);
    create_randn_matrix(model->W2, OUTPUT_SIZE * HIDDEN_LAYER_SIZE);
    create_randn_matrix(model->B2, OUTPUT_SIZE);
}

void free_model(Model *model){
    free(model->W1);
    free(model->B1);
    free(model->W2);
    free(model->B2);
}

void train_model(Model *model, double *X_train, double *y_train, 
             const int train_size, const int epochs, const double learning_rate){
    
    double Z1[HIDDEN_LAYER_SIZE], A1[HIDDEN_LAYER_SIZE];
    double Z2[OUTPUT_SIZE];

    for (int iter = 0; iter < epochs; iter++)
    {
        double error = 0.0;

        // forward pass for input to hidden (1st dense)
        
        dense_forward(X_train, model->W1, model->B1, Z1, INPUT_SIZE, HIDDEN_LAYER_SIZE);
        
        // activation function on hidden layer (1st activation)
        for (int i = 0; i < HIDDEN_LAYER_SIZE; i++)
        {
            A1[i] = ReLU(Z1[i]);
        }

        // forward pass for the hidden layer (after activaiton) to output layer (2nd dense)

        dense_forward(A1, model->W2, model->B2, Z2, HIDDEN_LAYER_SIZE, OUTPUT_SIZE);

        // forward pass ready ---

        // computing loss w.r.t the output (prep for backward pass)

        error = mse(y_train, Z2, train_size);
        printf("Epoch: %d: Loss: %f", iter, error);

        //backprop

        double dCdZ2[OUTPUT_SIZE];

        //computing dCdY
        mse_deriv(y_train, Z2, dCdZ2, train_size); 

        // output to hidden backprop
        double dCdA1[HIDDEN_LAYER_SIZE];
        dense_backward(dCdZ2, dCdA1, A1, model->W2, model->B2, 
                    HIDDEN_LAYER_SIZE, OUTPUT_SIZE, learning_rate);

        double dCdZ1[HIDDEN_LAYER_SIZE];
        for (int i = 0; i < HIDDEN_LAYER_SIZE; i++)
        {
            dCdZ1[i] = dCdA1[i] * (Z1[i]);
        }

        double dCdX[INPUT_SIZE];
        dense_backward(dCdZ1, dCdX, X_train, model->W1, model->B1, 
                    INPUT_SIZE, HIDDEN_LAYER_SIZE, learning_rate);
    }
}

