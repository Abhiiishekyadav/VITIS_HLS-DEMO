#include "ap_int.h"
#include "ap_fixed.h"
#include <math.h>
#define MAX 5
typedef ap_fixed<8,6> float8;
typedef ap_int<8> int8;

float8 sigmoid(float8 x) {
	float scale_down = x/MAX;
	float y = exp(scale_down)/(1+exp(scale_down)) ;
    return y ;
}

void nn(float8 X1, float8 X2, float8 X3, int8 n, float8 *c1, float8 *c2, float8 *c3) {
    #pragma HLS INTERFACE s_axilite port=n bundle=CTRL_BUS
    #pragma HLS INTERFACE s_axilite port=return bundle=CTRL_BUS
    #pragma HLS INTERFACE axis port=X1
    #pragma HLS INTERFACE axis port=X2
    #pragma HLS INTERFACE axis port=X3
    #pragma HLS INTERFACE axis port=c1
    #pragma HLS INTERFACE axis port=c2
    #pragma HLS INTERFACE axis port=c3

    // Input layer
    float8 I_1_1 = X1;
    float8 I_2_1 = X2;
    float8 I_3_1 = X3;
    for (int i = 0; i < n-1; i++) {
            I_1_1 += X1;
            I_2_1 += X2;
            I_3_1 += X3;
        }

    // First hidden layer
    float8 H_1_2 = sigmoid(I_1_1 + I_2_1 + I_3_1);
    float8 H_2_2 = sigmoid(I_1_1 + I_2_1 + I_3_1);
    float8 H_3_2 = sigmoid(I_1_1 + I_2_1 + I_3_1);

    // Second hidden layer
    float8 H_1_3 = sigmoid(H_1_2 + H_2_2 + H_3_2);
    float8 H_2_3 = sigmoid(H_1_2 + H_2_2 + H_3_2);
    float8 H_3_3 = sigmoid(H_1_2 + H_2_2 + H_3_2);

    // Third hidden layer
    float8 H_1_4 = sigmoid(H_1_3 + H_2_3 + H_3_3);
    float8 H_2_4 = sigmoid(H_1_3 + H_2_3 + H_3_3);
    float8 H_3_4 = sigmoid(H_1_3 + H_2_3 + H_3_3);

    // Output layer
    float8 O_1_5 = sigmoid(H_1_4 + H_2_4 + H_3_4);
    float8 O_2_5 = sigmoid(H_1_4 + H_2_4 + H_3_4);
    float8 O_3_5 = sigmoid(H_1_4 + H_2_4 + H_3_4);

    // Assign output values
    *c1 = O_1_5;
    *c2 = O_2_5;
    *c3 = O_3_5;
}
