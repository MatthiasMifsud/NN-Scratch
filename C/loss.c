double mse(const double *y_true, const double *y_pred, const int size){
    double mse = 0.0;
    for (int i = 0; i < size; i++)
    {
        double mse_diff = y_pred[i] - y_true[i];
        mse += mse_diff * mse_diff;
    }

    return mse / (double)size;
}

void mse_deriv(const double *y_true, const double *y_pred, 
                double *dCdY, const int size){
    for (int i = 0; i < size; i++)
    {
        dCdY[i] = 2.0 * (y_pred[i] - y_true[i]) / (double)size;
    }
}
