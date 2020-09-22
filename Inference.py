import numpy as np
import torch
import os
import argparse
import pickle
from PIL import Image
import sys
import math

parser = argparse.ArgumentParser(description="CNN for MNIST or basic classification tasks")
parser.add_argument("--model_save_path",  type=str, default="/Users/varunbabbar/Desktop/CNN_Architecture",help="The directory to save the model to (including model name)")
parser.add_argument("--input_image", type=str, default="", help="Full path to the input image that is being classified")
opt = parser.parse_args()

model_save_path = opt.model_save_path
input_image_path = opt.input_image

X = np.array(Image.open(input_image_path))

if len(X.shape) == 2:
    X = torch.Tensor(np.expand_dims(X, axis=0))

sigma = []
with open(model_save_path, 'rb') as f:
    architecture = pickle.load(f)

def im2col(X,kernel,stride,im_needed = True,shape_specified = False):
        """Assuming X and the kernel are 2D inputs
            We transform the input matrix X into a matrix Ω(x) such that the columns correspond to the
            set of elements that will be multiplied by the kernel in each sliding convolution operation
            Given x -> Ω(x), conv2d(x,kernel) = Ω(x)' * kernel where ' denotes a transpose operation
        """
        try:
            X = X.detach().numpy()
        except:
            pass
        X = np.array(X)
        if not shape_specified: # If the kernel shape is not specified, then it is assumed that the argument kernel is a weight matrix
            kernel_shape = kernel.shape
        else: # If the kernel shape is specified, then it is assumed that the argument is the shape of the weight, not the weight itself
            kernel_shape = kernel

        img_shape = np.shape(X)
        a = np.empty((kernel_shape[0]*kernel_shape[1]*kernel_shape[2]))
        output_matrix = np.reshape(a,(kernel_shape[0]*kernel_shape[1]*kernel_shape[2],1))
        q = 0
        output_size1 = math.floor((img_shape[1] - kernel_shape[1])/(stride)) + 1
        output_size2 = math.floor((img_shape[2] - kernel_shape[2])/(stride)) + 1
        if im_needed:
            indicator_dict = {}            # Map (i,j,k) to (p,q)  (i,j,k): (p,q)
            coord_matrix = np.zeros((img_shape[0],img_shape[1],img_shape[2]),dtype='object')
            for k in range(0,img_shape[2]):
                 for j in range(0,img_shape[1]):
                    for i in range(0,img_shape[0]):
                        coord_matrix[i,j,k] = (i,j,k)
                    
            for k in range(0,img_shape[2],stride):
                 for j in range(0,img_shape[1],stride):
                    for i in range(0,img_shape[0],stride):
                            try:
                                vals = X[i:i+kernel_shape[0],j:j+kernel_shape[1],k:k+kernel_shape[2]]
                                vals = np.reshape(vals,(int(np.shape(vals)[2]*np.shape(vals)[1]*np.shape(vals)[0]),1))
                                output_matrix = np.hstack((output_matrix,vals))
                                indices = coord_matrix[i:i+kernel_shape[0],j:j+kernel_shape[1],k:k+kernel_shape[2]]

                                indices = np.reshape(indices,(int(np.shape(indices)[2]*np.shape(indices)[1]*np.shape(indices)[0])))
                                coords = [(p,q) for p in range(len(indices))]
                                d = dict(zip(coords,list(indices)))
                                indicator_dict.update(d)
                                q+=1

                            except:
                                 continue
                                    
            reversed_indicator_dict = dict()
            for k,v in indicator_dict.items():
                reversed_indicator_dict.setdefault(v, []).append(k) # Reversed mapping now in order to
                           
            # I have created an indicator dictionary that maps the (p,q) coordinates in the im2col representation
            # of the input to the (i,j,k) coordinates of the actual input
            # This will be used while calculating gradients in backprop
            output_matrix = np.delete(output_matrix,0,1)
            output_matrix = torch.FloatTensor(output_matrix)
            return output_matrix,reversed_indicator_dict
        else:
            for k in range(0,img_shape[2],stride):
                 for j in range(0,img_shape[1],stride):
                    for i in range(0,img_shape[0],stride):
                            try:
                                vals = X[i:i+kernel_shape[0],j:j+kernel_shape[1],k:k+kernel_shape[2]]
                                vals = np.reshape(vals,(int(np.shape(vals)[2]*np.shape(vals)[1]*np.shape(vals)[0]),1))
                                output_matrix = np.hstack((output_matrix,vals))
                            except:
                                continue
            output_matrix = np.delete(output_matrix,0,1)
            output_matrix = torch.FloatTensor(output_matrix)
            return output_matrix,-1

def cross_entropy(y_pred,y):
    """ Cross entropy loss for classification purposes"""
    epsilon = 0.001 # To prevent overflow and ensure numerical stability
    return sum(-y*np.log(y_pred+epsilon))

def sigmoid(X):
    """ Sigmoid Activation Function"""
    X[X < -300] = -300 # For stability
    X = X.detach().numpy()
    X = torch.FloatTensor((1/(1+(np.exp(-X)))))
    return X
    
def softmax(y):
    """ Simple softmax activation with enhanced stability
    """
#     y = y.squeeze()
    epsilon = 0.001
    y = y.detach().numpy()
    y[y > 400] = 400 # For stability to prevent overflow
    denominator = epsilon + sum(np.exp(y)) # Further stability to prevent overflow
    numerator = np.exp(y)
    softmax = numerator / denominator
    return torch.Tensor(softmax)
    
def relu(x):
    """ ReLU Activation Function
    """
    return x.clamp_min(0.)
    
def maxpool_im2col(X,kernel_shape,stride):

    """ Converts the input into a image-to-column representation
    where each column is the window of elements over which the maximum is taken
    """

    output_size1 = math.floor((X.shape[1] - kernel_shape[1])/(stride)) + 1
    output_size2 = math.floor((X.shape[2] - kernel_shape[2])/(stride)) + 1

    if len(kernel_shape) == 2:
        kernel_shape = torch.reshape(kernel_shape,(1,kernel_shape[0],kernel_shape[1]))

    im = {}

    for i in range(X.shape[0]):
        Xi = X[i,:,:]
        Xi = torch.reshape(Xi,(-1,Xi.shape[0],Xi.shape[1]))
        X_im2col,imi = im2col(Xi,kernel_shape,stride,im_needed = True,shape_specified = True)
        if i == 0:
            X_im2c = torch.zeros((X.shape[0],X_im2col.shape[0],X_im2col.shape[1]))
            output = torch.zeros((X.shape[0],X_im2col.shape[1]))
        X_im2c[i,:,:] = X_im2col
    # Equivalent indicator dictionary representation created for the purposes of backpropagation

    return X_im2c

def maxpool(X,kernel_shape,stride,*args,return_architecture = False):

    """ Applies a max-pool of input features.
    This ensures a more dense representation of features, which can be computationally efficient
    because of lower dimensionality. It also helps in capturing more important features.
    """
    
    if len(kernel_shape) == 2:
        kernel_shape = torch.reshape(kernel_shape,(1,kernel_shape[0],kernel_shape[1]))
    output_size1 = math.floor((X.shape[1] - kernel_shape[1])/(stride)) + 1
    output_size2 = math.floor((X.shape[2] - kernel_shape[2])/(stride)) + 1
    if return_architecture:
        im = {}
        imx = {} # inverted im
    for i in range(X.shape[0]):
        Xi = X[i,:,:]
        Xi = torch.reshape(Xi,(-1,Xi.shape[0],Xi.shape[1]))
        if return_architecture:
            X_im2col,imi = im2col(Xi,kernel_shape,stride,im_needed = True,shape_specified = True)
        else:
            X_im2col,imi = im2col(Xi,kernel_shape,stride,im_needed = False,shape_specified = True)
        if i == 0:
            X_im2c = torch.zeros((X.shape[0],X_im2col.shape[0],X_im2col.shape[1]))
            output = torch.zeros((X.shape[0],X_im2col.shape[1]))
        X_im2c[i,:,:] = X_im2col
        if return_architecture:
            im[i] = imi
            imx[i] = {value: key for key in imi for value in imi[key]}
        for j in range(X_im2col.shape[1]):
            output[i,j] = max(X_im2col[:,j])
    output_shape = (X.shape[0],output_size1,output_size2)
    output = torch.reshape(output,output_shape)

    if return_architecture:
        return output,output_shape,X_im2c,im,imx
    else:
        return output
        
        
def BatchNorm(X):
    # (X - mu) / sigma -> Have to implement trainable parameters gamma and beta on this

    """ Simple non-trainable instance-normalisation of the input.
    """
    
    epsilon = 0.001  # To prevent overflow and ensure numerical stability
    bn = (X - torch.mean(X)) / (torch.std(X)+epsilon)
    sigma.append(torch.std(X)+epsilon)
    return bn

def flatten_and_dense(X,out_channels,*args,activation = 'relu', initialise_weights = False):
    """ Flattens the input and outputs a fully connected dense layer after applying an activation function
        Extra args correspond to weights and biases input"""
    shape = X.shape
    X = torch.reshape(X,(-1,1)) # Flatten
    if initialise_weights:
        weights = torch.Tensor(np.random.uniform(-0.01,0.01, size = (out_channels,len(X))))
        weights.requires_grad = False
        bias = torch.Tensor(np.random.uniform(-0.01,0.01, size = (out_channels,1)))
    else:
        weights = args[0]
        bias = args[1]
    if activation == 'sigmoid':
        output = sigmoid(weights.mm(X) + bias)
    elif activation == 'relu':
        output = relu(weights.mm(X) + bias)
    else:
        output = weights.mm(X) + bias # No activation applied -> Typically done before the softmax
    if not initialise_weights:
        
        return output
    else:
        output_shape = output.shape
        return output,weights,bias,output_shape



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

y_pred = forward_pass(X,architecture)
sys.stdout.write("\n")
sys.stdout.write("Predicted Number: ")

sys.stdout.write(str(int(np.argmax(y_pred))))

sys.stdout.write("\n")
sys.stdout.write("\n")
