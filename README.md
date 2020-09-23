# An implementation of a Vanilla CNN from scratch 

## Table of contents
* [General info](#general-info)
* [Dependencies](#dependencies)
* [Training Procedure](#training-procedure)
* [Inference Procedure](#inference-procedure)

## General info  
A customisable Convolutional Neural Network created from scratch using numpy arrays and torch tensors. Work still in progress, but current features include:
1) Activation functions: ReLU, Sigmoid, Softmax
2) Operations possible: Conv2d, Conv3d, Maxpool, BatchNorm, Flatten, Fully Connected Layer
3) Optimisers: Adam, SGD
4) Loss: Cross Entropy, L2, L1
5) Backprop across all operations
6) Max accuracy of 92% on MNIST with 8 layer 10 channel conv_bn_relu and maxpool architecture
7) Possible to customise architecture
8) Fast enough for basic classification tasks -> Takes a couple of hours on CPU to train 20000 examples and validate 5000 examples of MNIST with the above architecture. Not tested on GPU yet.

In the future:
1) Residual connections (Hopefully it should be possible to create a ResNet 13 from scratch)
2) Expanded scope beyond classification -> More along the lines of generation (Maybe a GAN?)
3) Improved speed
4) More optimizers including SGD with momentum, Adagrad, and RMSProp

## Dependencies
The modules are created with:
* [PyTorch 1.6](https://pytorch.org/get-started/locally/)
* Python 3.7
* [Numpy 1.19.2](https://pypi.org/project/numpy/)


## Training Procedure
Training involves running the script CNN.py. Here is the syntax for calling CNN.py through command line: 
 ```
   usage: CNN.py [-h] [--batchSize BATCHSIZE] [--nEpochs NEPOCHS] [--lr LR]
              [--step STEP] [--model_save_path MODEL_SAVE_PATH]
              [--lr_multiplier LR_MULTIPLIER] [--optimizer OPTIMIZER]
              [--num_examples NUM_EXAMPLES] [--val_examples VAL_EXAMPLES]

CNN for MNIST or basic classification tasks

optional arguments:
  -h, --help            show this help message and exit
  --batchSize BATCHSIZE
                        training batch size
  --nEpochs NEPOCHS     number of epochs to train for
  --lr LR               Learning Rate. Default=1e-4
  --model_save_path MODEL_SAVE_PATH
                        Full Path to trained Model (including model name)
  --lr_multiplier LR_MULTIPLIER
                        The decay factor of the learning rate. A more negative
                        factor will cause a greater exponential decay of the
                        learning rate over training epochs
  --optimizer OPTIMIZER
                        Optimizer for training: adam or sgd
  --num_examples NUM_EXAMPLES
                        Number of training examples
  --val_examples VAL_EXAMPLES
                        Number of test examples
   ```  


 ### Example usage:
 ```
 python3 CNN.py --batchSize 32 --nEpochs 10 --lr 0.01 --model_save_path /Users/varunbabbar/Desktop/MNIST_CNN --optimizer adam --num_examples 10000 --val_examples 2500
 ```
 
 ## Inference Procedure
Inference involves running the script Inference.py. This assumes you have a trained model and input image saved in your local filesystem. Here is the syntax for calling Inference.py through command line:
 ```
   usage: CNN_inference.py [-h] [--model_save_path MODEL_SAVE_PATH]
                        [--input_image INPUT_IMAGE]

CNN for MNIST or basic classification tasks

arguments:
  -h, --help            show this help message and exit
  --model_save_path MODEL_SAVE_PATH
                        The directory to save the model to (including model
                        name)
  --input_image INPUT_IMAGE
                        Full path to the input image that is being classified

   ```  


 ### Example usage:
 ```
 python3 CNN_inference.py --model_save_path /Users/varunbabbar/Desktop/CNN_Architecture --input_image /Users/varunbabbar/Desktop/4.jpg
 
 Predicted Number: 4
 ```
 
 






