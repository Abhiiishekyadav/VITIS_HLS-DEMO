// cnn.h

#ifndef CNN_H
#define CNN_H

#include <cmath>
#include <random>
 #include "mnist_weights.h"

double relu(double x);
void addNoise(double input[], double stddev, int input_size);
void convolution(const double flattenedImage[], const double kernels[], double output[], int imageWidth, int imageHeight, int imageDepth, int k_h, int k_w, const double biases[], int numKernels,char a_f);
void maxPooling(const double image[], double output[], int imageWidth, int numChannels, int pool_size);
void fullyConnectedLayer(const double input[], double output[], const double weights[], const double bias[], int inputSize, int outputSize,char a_f);
void CNN(const double flattenedImage[], int imageWidth, int imageHeight, double output[]);

#endif // CNN_H
