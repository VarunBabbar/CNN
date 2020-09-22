import numpy as np
import torch
import os
import argparse

parser = argparse.ArgumentParser(description="CNN for MNIST or basic classification tasks")
parser.add_argument("--model_save_path",  type=str, default="/Users/varunbabbar/Desktop",help="The directory to save the model to (including model name)")





def forward_pass(X,architecture):
    
    """Performs a forward pass over the neural network and stores the
    resulting weights and features in the architecture dictionary.
    """
    
   
    architecture['layer1'][0] = X
    kernel_shape1 = architecture['layer1'][7]
    stride1 = architecture['layer1'][8]
    if kernel_shape1 is not None and not isinstance(kernel_shape1,int):
        X_input_1_im2col,imX = im2col(X,kernel_shape1,stride1,im_needed = False, shape_specified = True)
        architecture['layer1'][4] = X_input_1_im2col
    else:
        architecture['layer1'][4] = None

    for layer in range(len(architecture)): # Feedforward from the first till the second last layer
            X_input,X_output,weightsi,biasi,X_input_1_im2col,imi,output_shapei,kernel_shapei,stridei,operationi,imx = architecture['layer{}'.format(layer+1)]

            if operationi == 'conv_bn_relu':
                conv_output = relu(BatchNorm(torch.t(X_input_1_im2col).mm(weightsi) + biasi))
                conv_output = torch.reshape(conv_output,output_shapei)
                architecture['layer{}'.format(layer+1)][1] = conv_output # resetting output as convolved shape
                if layer != len(architecture) - 1:
                    architecture['layer{}'.format(layer+2)][0] = conv_output # resetting intput of next layer as convolved shape
                    kernel_shapei__1 = architecture['layer{}'.format(layer+2)][7]
                    stridei__1 = architecture['layer{}'.format(layer+2)][8]
                    operationi__1 = architecture['layer{}'.format(layer+2)][9]
                    if kernel_shapei__1 is not None and not isinstance(kernel_shapei__1,int):
                        if operationi__1 == 'maxpool':
                            architecture['layer{}'.format(layer+2)][4] = maxpool_im2col(conv_output,kernel_shapei__1,stridei__1)
                        else:
                            architecture['layer{}'.format(layer+2)][4],imX = im2col(conv_output,kernel_shapei__1,stridei__1,im_needed = False, shape_specified = True)
                        # resetting input im2col of next layer as the im2col of the output of this layer
                    else:
                        architecture['layer{}'.format(layer+2)][4] = None
            elif operationi == 'conv_relu':
                conv_output = relu(torch.t(X_input_1_im2col).mm(weightsi) + biasi)
                conv_output = torch.reshape(conv_output,output_shapei)
                architecture['layer{}'.format(layer+1)][1] = conv_output # resetting output as convolved shape
                if layer != len(architecture) - 1:
                    architecture['layer{}'.format(layer+2)][0] = conv_output # resetting intput of next layer as convolved shape
                    kernel_shapei__1 = architecture['layer{}'.format(layer+2)][7]
                    stridei__1 = architecture['layer{}'.format(layer+2)][8]
                    operationi__1 = architecture['layer{}'.format(layer+2)][9]
                    if kernel_shapei__1 is not None and not isinstance(kernel_shapei__1,int):
                        if operationi__1 == 'maxpool':
                            architecture['layer{}'.format(layer+2)][4] = maxpool_im2col(conv_output,kernel_shapei__1,stridei__1)
                        else:
                            architecture['layer{}'.format(layer+2)][4],imX = im2col(conv_output,kernel_shapei__1,stridei__1,im_needed = False, shape_specified = True)
                        # resetting input im2col of next layer as the im2col of the output of this layer
                    else:
                        architecture['layer{}'.format(layer+2)][4] = None
            elif operationi == 'conv_bn_sigmoid':
                conv_output = sigmoid(BatchNorm(torch.t(X_input_1_im2col).mm(weightsi) + biasi))
                conv_output = torch.reshape(conv_output,output_shapei)
                architecture['layer{}'.format(layer+1)][1] = conv_output # resetting output as convolved shape
                if layer != len(architecture) - 1:
                    architecture['layer{}'.format(layer+2)][0] = conv_output # resetting intput of next layer as convolved shape
                    kernel_shapei__1 = architecture['layer{}'.format(layer+2)][7]
                    stridei__1 = architecture['layer{}'.format(layer+2)][8]
                    operationi__1 = architecture['layer{}'.format(layer+2)][9]
                    if kernel_shapei__1 is not None and not isinstance(kernel_shapei__1,int):
                        if operationi__1 == 'maxpool':
                            architecture['layer{}'.format(layer+2)][4] = maxpool_im2col(conv_output,kernel_shapei__1,stridei__1)
                        else:
                            architecture['layer{}'.format(layer+2)][4],imX = im2col(conv_output,kernel_shapei__1,stridei__1,im_needed = False, shape_specified = True)
                        # resetting input im2col of next layer as the im2col of the output of this layer
                    else:
                        architecture['layer{}'.format(layer+2)][4] = None
            elif operationi == 'conv_sigmoid':
                conv_output = sigmoid(torch.t(X_input_1_im2col).mm(weightsi) + biasi)
                conv_output = torch.reshape(conv_output,output_shapei)
                architecture['layer{}'.format(layer+1)][1] = conv_output # resetting output as convolved shape
                if layer != len(architecture) - 1:
                    architecture['layer{}'.format(layer+2)][0] = conv_output # resetting intput of next layer as convolved shape
                    kernel_shapei__1 = architecture['layer{}'.format(layer+2)][7]
                    stridei__1 = architecture['layer{}'.format(layer+2)][8]
                    operationi__1 = architecture['layer{}'.format(layer+2)][9]
                    if kernel_shapei__1 is not None and not isinstance(kernel_shapei__1,int):
                        if operationi__1 == 'maxpool':
                            architecture['layer{}'.format(layer+2)][4] = maxpool_im2col(conv_output,kernel_shapei__1,stridei__1)
                        else:
                            architecture['layer{}'.format(layer+2)][4],imX = im2col(conv_output,kernel_shapei__1,stridei__1,im_needed = False, shape_specified = True)
                        # resetting input im2col of next layer as the im2col of the output of this layer
                    else:
                        architecture['layer{}'.format(layer+2)][4] = None
            elif operationi == 'maxpool':
                maxpool_output = maxpool(X_input,kernel_shapei,stridei)

                maxpool_output = torch.reshape(maxpool_output,output_shapei)

                if layer != len(architecture) - 1:
                    architecture['layer{}'.format(layer+2)][0] = maxpool_output
                    kernel_shapei__1 = architecture['layer{}'.format(layer+2)][7]
                    stridei__1 = architecture['layer{}'.format(layer+2)][8]
                    if kernel_shapei__1 is not None and not isinstance(kernel_shapei__1,int):
                        architecture['layer{}'.format(layer+2)][4],imX = im2col(maxpool_output,kernel_shapei__1,stridei__1,im_needed = False, shape_specified = True)
                    else:
                        architecture['layer{}'.format(layer+2)][4] = None
            elif operationi == 'flatten_dense_relu':
                # kernel_shapei in this case refers to the output channels: stride for dense layer will be None
                output = flatten_and_dense(X_input,kernel_shapei,weightsi,biasi,activation = 'relu',initialise_weights = False)
                architecture['layer{}'.format(layer+1)][1] = output
                if layer != len(architecture) - 1:
                    architecture['layer{}'.format(layer+2)][0] = output
            elif operationi == 'flatten_dense_none':
                # kernel_shapei in this case refers to the output channels: stride for dense layer will be None
                output = flatten_and_dense(X_input,kernel_shapei,weightsi,biasi,activation = 'none',initialise_weights = False)
                architecture['layer{}'.format(layer+1)][1] = output
                if layer != len(architecture) - 1:
                    architecture['layer{}'.format(layer+2)][0] = output
            elif operationi == 'flatten_dense_sigmoid':
                # kernel_shapei in this case refers to the output channels: stride for dense layer will be None
                output = flatten_and_dense(X_input,kernel_shapei,weightsi,biasi,activation = 'sigmoid',initialise_weights = False)
                architecture['layer{}'.format(layer+1)][1] = output
                if layer != len(architecture) - 1:
                    architecture['layer{}'.format(layer+2)][0] = output
            elif operationi == 'softmax':
                Xin = architecture['layer{}'.format(layer+1)][0]
                output = softmax(Xin).squeeze()
                architecture['layer{}'.format(layer+1)][1] = output
            if layer == len(architecture) - 1:
                y_pred = architecture['layer{}'.format(len(architecture))][1]
                
    return y_pred
