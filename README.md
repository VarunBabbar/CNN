# CNN
A customisable Convolutional Neural Network created from scratch using numpy arrays and torch tensors. Work still in progress, but current features include:
1) Activation functions: ReLU, Sigmoid, Softmax
2) Operations possible: Conv2d, Conv3d, Maxpool, BatchNorm, Flatten, Fully Connected Layer
3) Optimisers: Adam, SGD
4) Loss: Cross Entropy, L2, L1
5) Backprop across all operations
6) Max accuracy of 85% on MNIST with 8 layer 10 channel conv_bn_relu and maxpool architecture
7) Possible to customise architecture
8) Fast enough for basic classification tasks -> Takes a couple of hours on CPU to train 10000 examples and validate 2500 examples of MNIST with the above architecture. Not tested on GPU yet.

In the future:
1) Residual connections (Hopefully it should be possible to create a ResNet 13 from scratch)
2) Expanded scope beyond classification -> More along the lines of generation (Something like a GAN)
3) Improved speed


