#include <stdlib.h>
#include <math.h>

double randn(const int ROW, const int COLUMN){
    
    static int haveSpare = 0;
    static double spare;

    if (haveSpare){
        haveSpare = 0;
    }

    haveSpare = 1;

    double u,v,s;
    
    do{
        u = (rand() / ((double)RAND_MAX)) * 2.0 - 1.0;
        v = (rand() / ((double)RAND_MAX)) * 2.0 - 1.0;

    }


    
    double matrix[ROW][COLUMN];
}


int main(void){

    double matrix[10][10] = randn(10, 10);

    return 0;
}