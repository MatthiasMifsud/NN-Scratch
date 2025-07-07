// model structure: input - hidden - relu - output

#include "model.h"
#include "data.h"

int main(void){
    Model model;

    init_param(&model);

    load_data(INPUT_SIZE, OUTPUT_SIZE);
    check_data(INPUT_SIZE, OUTPUT_SIZE);
    const int epochs = 100;
    const double learning_rate = 0.1;

    train_model(&model, data.X_train, data.y_train_one_hot, 
                train_size, epochs, learning_rate);

    free_model(&model);
    free_data();
    return 0;
}

//compile 
//gcc -o nn main.c data.c model.c activation.c fullyconnected.c loss.c 
//./nn
    