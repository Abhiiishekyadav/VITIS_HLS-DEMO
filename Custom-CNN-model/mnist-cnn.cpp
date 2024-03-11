#include "cnn.h"
#pragma inline
double relu(double x)
{
    return (x < 0) ? 0 : x;
}

//// Function to add noise to a 3D matrix
void addNoise(double input[], double stddev, int input_size) {
    double mean = 0.0;

    // Seed for basic linear congruential generator
    unsigned int seed = 42; // You can use any initial value

    #pragma HLS loop unroll factor=2 // Adjust the unroll factor based on your design and target FPGA

    for (int i = 0; i < input_size; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 1
        // Basic linear congruential generator
        seed = (seed * 1664525u + 1013904223u) & 0xFFFFFFFF;
        double random_value = static_cast<double>(seed) / UINT_MAX;

        double noise = mean + stddev * (2 * random_value - 1);
        input[i] += noise;
    }
}


void softmax(const double input[], int size, double output[]) {
     double max_val = input[0];
    for (int i = 1; i < size; ++i) {
    	#pragma HLS LOOP_TRIPCOUNT min = 1 max = 1
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }

    double sum_exp = 0.0;

    for (int i = 0; i < size; ++i) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 1
        sum_exp += std::exp(input[i] - max_val);
    }

    for (int i = 0; i < size; ++i) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 1
        output[i] = std::exp(input[i] - max_val) / sum_exp;
        }

}


void convolution(const double flattenedImage[], const double kernels[], double output[], int imageWidth, int imageHeight, int imageDepth, int k_h, int k_w, const double biases[], int numKernels , char a_f)
{
    int output_h = imageHeight - k_h + 1;
    int output_w = imageWidth - k_w + 1;

   #pragma HLS UNROLL factor=2 // Adjust the unroll factor based on your design and target FPGA

    for (int k = 0; k < numKernels; ++k)
    {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 1
   #pragma HLS UNROLL factor=2 // Adjust the unroll factor based on your design and target FPGA
        for (int i = 0; i < output_h; ++i)
        {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 1
      #pragma HLS UNROLL factor=2 // Adjust the unroll factor based on your design and target FPGA
            for (int j = 0; j < output_w; ++j)
            {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 1
              #pragma HLS PIPELINE II=2 // Adjust the pipeline II factor based on your design and target FPGA
                output[k * output_h * output_w + i * output_w + j] = biases[k]; // Initialize output with bias for the current kernel

                for (int kd = 0; kd < imageDepth; ++kd)
                {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 1
                    for (int ki = 0; ki < k_h; ++ki)
                    {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 1
                        for (int kj = 0; kj < k_w; ++kj)
                        {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 1
                            output[k * output_h * output_w + i * output_w + j] +=
                                flattenedImage[(j + kj) + (i + ki) * imageWidth + kd * imageHeight * imageWidth] * kernels[k * (k_h * k_w * imageDepth) + ki * k_w + kd * k_h * k_w + kj];
                        }
                    }
                }
                if(a_f == 'R') {
                    output[k * output_h * output_w + i * output_w + j] = relu(output[k * output_h * output_w + i * output_w + j]);
                }
            }
        }
    }
}
void maxPooling(const double image[], double output[], int imageWidth, int numChannels, int pool_size) {
    int i_h = (imageWidth);
    int i_w = i_h;
    int output_h = i_h / pool_size;
    int output_w = i_w / pool_size;

    for (int c = 0; c < numChannels; ++c) {
        for (int i = 0; i < output_h; ++i) {
            for (int j = 0; j < output_w; ++j) {
                double max_val = 0.0;
                for (int pi = 0; pi < pool_size;pi+=2) {
                    for (int pj = 0; pj < pool_size; pj+=2) {
                        max_val = fmax(max_val, image[c * (i_h * i_w) + (i * pool_size + pi) * i_w + (j * pool_size + pj)]);
                    }
                }
                output[c * (output_h * output_w) + i * output_w + j] = max_val;
            }
        }
    }
}


void fullyConnectedLayer(const double input[], double output[], const double weights[], const double bias[], int inputSize, int outputSize, char a_f) {

	for (int i = 0; i < outputSize; ++i) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 1
        output[i] = bias[i];
        for (int j = 0; j < inputSize; ++j) {
#pragma HLS LOOP_TRIPCOUNT min = 1 max = 1
            output[i] += input[j] * weights[i * inputSize + j];
        }

        if (a_f=='R') {
            output[i] = (output[i] < 0) ? 0 : output[i];
        }
    }
}


void CNN(const double flattenedImage[], int imageWidth, int imageHeight, double output[]) {
    #pragma HLS INTERFACE s_axilite port=imageWidth bundle=control
    #pragma HLS INTERFACE s_axilite port=imageHeight bundle=control
    #pragma HLS INTERFACE s_axilite port=flattenedImage bundle=control
    #pragma HLS INTERFACE s_axilite port=output bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    // Convolutional layer1
    int num_kernels1 = 84;
    int kernelWidth1 = 3;
    int kernelHeight1 = 3;
    int imagedepth1 = 1;
    char a_f = 'R';
    double conv1Output[26*26*84];
    convolution(flattenedImage, convolution1_weights, conv1Output, imageWidth, imageHeight, imagedepth1, kernelHeight1, kernelWidth1, convolution1_bias, num_kernels1, a_f);
    addNoise(conv1Output, 0.078735, 26*26*84);

    // Convolutional layer2
    int num_kernels2 = 32;
    int kernelWidth2 = 3;
    int kernelHeight2 = 3;
    int imagedepth2 = 84;
    double conv2Output[24*24*32];
    convolution(conv1Output, convolution2_weights, conv2Output, 26, 26, imagedepth2, kernelHeight2, kernelWidth2, convolution2_bias, num_kernels2, a_f);
    addNoise(conv2Output, 0.046218, 24*24*32);

    // convulation 3
    double conv3Output[22*22*132];
    convolution(conv2Output, convolution3_weights, conv3Output, 24, 24, 32, 3, 3, convolution3_bias, 132, a_f);
    addNoise(conv3Output, 0.089512, 22*22*132);

    // convulation 4
    double conv4Output[20*20*64];
    convolution(conv3Output, convolution4_weights, conv4Output, 22, 22, 132, 3, 3, convolution4_bias, 64, a_f);
    addNoise(conv4Output, 0.027931, 20*20*64);

    // convolution 5
    double conv5Output[18*18*118];
    convolution(conv4Output, convolution5_weights, conv5Output, 20, 20, 64, 3, 3, convolution5_bias, 118, a_f);
    addNoise(conv5Output, 0.053142, 18*18*118);

    // fully connected layer.
    char a_f2 = 'S';
    double fully1output[10];
    fullyConnectedLayer(conv5Output, fully1output, dense1_weights, dense1_bias, 38232, 10, a_f2);

    char a_f3 = 'S';
    if (a_f3 == 'S') {
        softmax(fully1output, 10, output);
    }
}


