// model structure: input - hidden - relu - output

#include "model.h"
#include "data.h"

int main(void){

    load_data(INPUT_SIZE, OUTPUT_SIZE);

    Model model;
    init_param(&model);

    free_data();
    free_model(&model);
}

//compile 
//gcc -o nn-scratch main.c data.c model.c activation.c fullyconnected.c loss.c 
//./nn-scratch