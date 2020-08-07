# CNN
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
2) Expanded scope beyond classification -> More along the lines of generation (Something like a GAN)
3) Improved speed


# SR_ResNet

README: Instructions for training and obtaining results for different loss functions

The codebase contains several different modules, which allows it to be fully customisable 
	
2) 
## Table of contents
* [General info](#general-info)
* [Usage](#usage)
* [Setup](#setup)

## General info
There are several modules here that enable the user to implement different loss functions / customize their own loss functions. These loss functions are primarily created in order to determine the properties of an ideal perceptual loss function that can be used in image to image translation networks.
	
## Usage
The modules are created with:
* Pytorch 1.6
* Python 3.7
	
## Instructions for use
To run this project, first clone it onto your local machine:

    ```
    $ git clone xxx

    ```
1) The DQ module can be imported as:
    ```python
    import DQ
    import torch
    dq = DQ(**kwargs)
    x = torch.rand(16,3,256,256)
    y = torch.rand(16,3,256,256)
    dq_loss = dq(x,y)
    dq_loss.backward()
    ```
    For a detailed description of input arguments, head over to the script DQ.py.

2) To implement any loss function in LUV colorspace, import the module LUV_Converter and instantiate it as luv = LUV_Converter()
     ```python
    import LUV_Converter
    luv = LUV_Converter()
    import torch
    x = torch.rand(16,3,256,256)
    y = torch.rand(16,3,256,256)
    x_luv = luv(x)
    y_luv = luv(y)
    
    loss_function = DQ()
    loss = loss_function(x_luv,y_luv)
    loss.backward()
    ```
3) The SpatialGradient module evaluates a loss function on the gradient of an image. This is of the form: loss = <img src="https://latex.codecogs.com/gif.latex?\beta(\frac{d(G(z))}{dx},\frac{dY}{dx})+\beta(\frac{d(G(z))}{dy},\frac{dY}{dy})+\lambda |G(z)-\mu_{G(z)}|" /> 
Here, 
Here, the user has the option of choosing to evaluate the loss on both the gradient magnitude and orientation. 
    An instance of SpatialGradient is of the form:
    
    ```python
    spatial_grad = SpatialGradient(loss_func = nn.MSELoss())
    ```
    Example Usage: 
    
    ```python
    import SpatialGradient
    import torch.nn as nn
    import torch
    mse_grad = SpatialGradient(loss_func = nn.MSELoss())
    x = torch.rand(16,3,256,256)
    y = torch.rand(16,3,256,256)
    loss = mse_grad(x,y)
    loss.backward()
    ```
    It is also possible to add any other pytorch loss functions as input arguments, including the modules in this repository.
     ```python
    import SpatialGradient
    import DQ
    import torch
    dq_grad = SpatialGradient(loss_func = DQ())
    x = torch.rand(16,3,256,256)
    y = torch.rand(16,3,256,256)
    loss = dq_grad(x,y)
    loss.backward()
    ```
    For a detailed description of input arguments, head over to the script SpatialGradient.py


4) The Multi Scale inherits nn.Module and is differentiable. It accepts a loss function as input and has several customisable parameters. Example Usage:
    ```python
    import MultiScale
    import torch.nn as nn
    import torch
    loss_function = MultiScale(loss_func=nn.MSELoss())
    x = torch.rand(16,3,256,256)
    y = torch.rand(16,3,256,256)
    loss = dq_grad(x,y)
    loss.backward()
    ```
    
    A MultiScale instance is of the form:
    ```python
    import MultiScale
    multi_scale_loss = MultiScale(loss_function = nn.MSELoss(), **kwargs)
    ```
    For a detailed description of input arguments, head over to the script MultiScale.py
    
   It is possible to have a MultiScale loss as an argument to the SpatialGradient function and vice versa. 
   a)
	   ```python
	    import MultiScale
	    import SpatialGradient
    	    import torch
	    dq_grad = SpatialGradient(loss_func = DQ())
	    loss_function = MultiScale(loss_func=dq_grad, **kwargs)
	    x = torch.rand(16,3,256,256)
	    y = torch.rand(16,3,256,256)
	    loss = dq_grad(x,y)
	    loss.backward()
	    ```
    b) 
	    ```python
	    import MultiScale
	    import SpatialGradient
	    import torch
	    dq_multi_scale = MultiScale(loss_func=DQ(), **kwargs)
	    dq_grad_multiscale = SpatialGradient(loss_func = dq_multi_scale)
	    x = torch.rand(16,3,256,256)
	    y = torch.rand(16,3,256,256)
	    loss = dq_grad(x,y)
	    loss.backward()
	    ```
    The difference between a) and b) is that in a), the image is transformed into an image pyramid, and for each image at each level and octave, DQ is applied on the gradient of the image at that scale. In b), the the gradient of the image is taken first and then the image gradient is transformed into a multi-scale pyramid (for DQ loss to be applied).
    
   
   
For a detailed description of input arguments, head over to the script MultiScale.py.

	⁃	Loss functions can be evaluated on VGG channels by importing the VGG_Normalized module. For example: vgg_loss = VGG_Normalized(loss_func=nn.MSELoss()). 


