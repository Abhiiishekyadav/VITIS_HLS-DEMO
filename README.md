This repository consist of DNN model implementation on fpga board (ZCU104):

Pre-trained model has been leverages and done using Python in the Google-Colab & related (.ipynb) file inside Software repository.

Basic building blocks of LeNet and AlexNet and custom CNN blocks are provided inside a different Hardware repository. Consist of different layers viz. linear, convolution , fully connected layer inside both models. The obtained high-level description of the pre-trained model and later, transform into RTL logic through the HLS process using the Vitis-HLS compiler and the obtained RTL are provided in the RTL and IP integration folder.

NOTE: 1) Each (.ipynb) file consists of software results, driver file and ARM-CPU result. Additionally, all C++ files consist of different AlexNet/LeNet/Custom CNN blocks code from scratch. 2) Pynq overlay file and other driver file also available in (.ipynb) file.

