#include <iostream>
#include <cmath>
#include "ap_int.h"
#include "ap_fixed.h"

#define MAX 5
typedef ap_fixed<8,6> float8;
typedef ap_int<8> int8;



void nn(float8 X1, float8 X2, float8 X3, int8 n, float8 *c1, float8 *c2, float8 *c3);

int main() {
    float8 X1 = 1.0;
    float8 X2 = 2.0;
    float8 X3 = 3.0;
    int8 n = 1;
    float8 c1, c2, c3;

    nn(X1, X2, X3, n, &c1, &c2, &c3);

    std::cout << "Output values: " << c1 << " " << c2 << " " << c3 << std::endl;

    return 0;
}
