import math
import numpy as np
import torch # Used only for matrix multiplication
from torch.autograd import Variable
import pandas as pd
import gc
import tensorflow as tf # Used only to acquire MNIST data
import argparse
import pickle
import os
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
X = []
device = torch.device("cpu")
# Probably an input of images (eg MNIST or something)
num_layers = 4
dtype = torch.float
torch.seed = 22
np.random.seed(22)
parser = argparse.ArgumentParser(description="CNN for MNIST or basic classification tasks")
parser.add_argument("--batchSize", type=int, default=32, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=7, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-2, help="Learning Rate. Default=1e-4")
parser.add_argument("--model_save_path",  type=str, default="/Users/varunbabbar/Desktop",help="The directory to save the model to (including model name)")
parser.add_argument("--lr_multiplier", type=float, default = -0.08, help="The decay factor of the learning rate. A more negative factor will cause a greater exponential decay of the learning rate over training epochs")
parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer for training: adam or sgd")
parser.add_argument("--num_examples", type=int, default=10000, help="Number of training examples")
parser.add_argument("--val_examples", type=int, default=2000, help="Number of test examples")
parser.add_argument("-f")
opt = parser.parse_args()
bs = opt.batchSize
epochs = opt.nEpochs
save_path = opt.model_save_path
optimizer = opt.optimizer
lr = opt.lr
lr_multiplier = opt.lr_multiplier
num_examples = opt.num_examples
val_examples = opt.val_examples
dropout = torch.nn.Dropout(p=0.2) # p = 0.3-0.5 is a good value
arr = []
epsilon = 1e-8
beta_1 = 0.9
beta_2 = 0.999
clip = 10000 # For gradient clipping
# torch.set_default_tensor_type('torch.cuda.FloatTensor') for GPU computations

# torch.set_printoptions(threshold=50000)  # Print full tensor for debugging purposes

# Convolutional Neural Network written from scratch for basic classification tasks (eg MNIST / Cats and Dogs)
# Can be used as a discriminator for a GAN as well

# To Do:
# Make train, predict, and conv functions part of a CNN_Engine class

def process_data(x_train,y_train,x_test,y_test,num_examples,val_examples):
    y = np.zeros((10,num_examples))

    for i in range(y.shape[1]):
        y[y_train[i],i] = 1 # MNIST labels
    y_input = y
    y = Variable(torch.FloatTensor(y))
    y.requires_grad = False

    X_inp = x_train[0:num_examples,:,:]
    X_inp = torch.FloatTensor(X_inp) # MNIST input

    y_val = np.zeros((10,val_examples))
    x_val = x_test[0:val_examples,:,:]
    x_val = torch.FloatTensor(x_val)
    for i in range(y_val.shape[1]):
        y_val[y_test[i],i] = 1
        y_val = torch.FloatTensor(y_val)
    return X_inp,y_input,x_val,y_val


def im2col(X,kernel,stride,im_needed = True,shape_specified = False):
        """Assuming X and the kernel are 2D inputs
            We transform the input matrix X into a matrix Ω(x) such that the columns correspond to the
            set of elements that will be multiplied by the kernel in each sliding convolution operation
            Given x -> Ω(x), conv2d(x,kernel) = Ω(x)' * kernel where ' denotes a transpose operation
        """
        X = X.detach().numpy()
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
    
def conv2D(null,channels,X,stride,kernel_shape,padding = False,initialize_weights = True,*args):

            """ Applies a 2D convolution over the input features.
            This is slightly different from the conv2D operation in PyTorch.
            In this operation there are C_out 3D kernels of shape (C_in,K_w,K_h)
            that are convolved with the input (shape = (C_in,W,H))
            

            """
            # filters = dimensionality of output space
            # If padding is enabled, we pad the input with zeros such that the input size
            # remains the same if weights with stride 1 are applied to the input
            if initialize_weights:
                kernel = np.random.normal(size = (kernel_shape[0],kernel_shape[1],kernel_shape[2]))*math.sqrt(1/(kernel_shape[0]*kernel_shape[1]*kernel_shape[2])) # Our input
                kernel = torch.FloatTensor(kernel)
                kernel.requires_grad = False
            else:
                kernel = args[0] # weights and bias must be given if initialise weights is disabled
                bias = args[1]
                kernel_shape = kernel.shape
            
            X = X.detach().numpy()
            if padding: # Can only pad during initialization -> weights and input shapes cannot change during feedforward and backpropagation
                if kernel_shape[1] % 2 == 0 and kernel_shape[2] % 2 == 0:
                    X = np.pad(X,((0,0),(math.floor(kernel_shape[1]/2)-1,math.floor(kernel_shape[1]/2)),(math.floor(kernel_shape[2]/2),math.floor(kernel_shape[2]/2)-1)), 'symmetric')
                elif kernel_shape[1] % 2 != 0 and kernel_shape[2] % 2 == 0:
                    X = np.pad(X,((0,0),(math.floor(kernel_shape[1]/2),math.floor(kernel_shape[1]/2)),(math.floor(kernel_shape[2]/2),math.floor(kernel_shape[2]/2)-1)), 'symmetric')
                elif kernel_shape[1] % 2 == 0 and kernel_shape[2] % 2 != 0:
                    X = np.pad(X,((0,0),(math.floor(kernel_shape[1]/2)-1,math.floor(kernel_shape[1]/2)),(math.floor(kernel_shape[2]/2),math.floor(kernel_shape[2]/2))), 'symmetric')
                else:
                    X = np.pad(X,((0,0),(math.floor(kernel_shape[1]/2),math.floor(kernel_shape[1]/2)),(math.floor(kernel_shape[2]/2),math.floor(kernel_shape[2]/2))), 'symmetric')
            
            X = torch.FloatTensor(X)
            
            img_shape = X.shape
            
            output_size1 = math.floor((img_shape[1] - kernel_shape[1])/(stride)) + 1
            output_size2 = math.floor((img_shape[2] - kernel_shape[2])/(stride)) + 1
            output_shape = [channels,output_size1,output_size2]
            
            X_im2col,im = im2col(X,kernel,stride)
            
            
            if initialize_weights:
                weight = torch.reshape(kernel,(kernel_shape[0]*kernel_shape[1]*kernel_shape[2],1))
                # weight consists of only one weight vector. But the dimensionality of output space has to be
                # num_filters. So we need to stack weight vectors horizontally and create num_filters number of
                # feature maps
                for i in range(channels-1):
                    weight2 = np.random.normal(size = (kernel_shape[0]*kernel_shape[1]*kernel_shape[2],1))*math.sqrt(1/(kernel_shape[0]*kernel_shape[1]*kernel_shape[2])) # Our input
                    weight2 = torch.FloatTensor(weight2)
                    weight2.requires_grad = False
                    weight = torch.cat((weight2, weight),1) # do this num_filters - 1 number of times
                conv_output = torch.t(X_im2col).mm(weight)
                bias = torch.Tensor(np.random.normal(size = conv_output.shape))
                conv_output += bias
                conv_output = torch.reshape(conv_output,(output_shape))
                return torch.nn.Parameter(conv_output), torch.nn.Parameter(weight),X_im2col,im, output_shape,bias
            else:
                # Since weights are already initialised, the relevant channels are already dictated in the architecture.
                # Therefore, conv output is just a matmul
                conv_output = torch.t(X_im2col).mm(kernel) + bias
                return torch.nn.Parameter(conv_output),X_im2col
                

     
    # kernel format = channels * rows * columns
    
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

def relu(x):

    """ ReLU Activation Function
    """
    
    return x.clamp_min(0.)

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

def BatchNorm(X): # (X - mu) / sigma -> Have to implement trainable parameters gamma and beta on this

    """ Simple non-trainable instance-normalisation of the input.
    """
    epsilon = 0.001  # To prevent overflow and ensure numerical stability
    bn = (X - torch.mean(X)) / (torch.std(X)+epsilon)
    sigma.append(torch.std(X)+epsilon)
    return bn

def conv_bn_relu(X,channels,stride,kernel_shape,padding,activation = 'sigmoid',initialize_weights = True,batchnorm = True,*args): # Refactoring conv_batchnorm_relu as one layer

    """ Applies a conv_batchnorm_relu layer on the input. The input is expected to be of shape
        (channels,width,height)
    """
    
    if initialize_weights: # no of channels in kernel must be the same as that in the input image,
        output_1,weights,X_im2col,im,output_shape,bias = conv2D(0,channels,X,stride,kernel_shape,padding,initialize_weights = True,*args) # conv
    else:
        output1,X_im2col = conv2D(0,channels,X,stride,kernel_shape,padding,initialize_weights = False,*args) # conv
    if batchnorm == True:
        output_1 = BatchNorm(output_1)
    if activation == 'sigmoid':
        X_1 = sigmoid(output_1)
    elif activation == 'relu':
        X_1 = relu(output_1)
    else:
        X_1 = output_1
    if initialize_weights:
        weights.requires_grad = False
        return torch.nn.Parameter(X_1),torch.nn.Parameter(weights),torch.nn.Parameter(X_im2col),im,output_shape,bias,activation
    else:
        return output1,X_im2col

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

def getIndexes(indicator_dictionary, val):

    ''' Get desired coordinates of val in an indicator matrix.
    val will be in the form of ijk cartesian coordinates'''
    
    try:
        return indicator_dictionary[val]
    except:
        return []
    
def residual(X1,X2,axis,concatenate=False):

    """Residual Layer applied between 2 feature maps"""
    
    if concatenate:
        output = torch.cat(X1,X2,axis = axis)
    else:
        output = torch.tensor(np.sum(X1,X2,axis = axis))
    return output

def num_params(architecture): #

    """Returns total number of trainable and non_trainable parameters in the network"""
    
    total_parameters = 0
    for layer in range(1,len(architecture)+1):
        weight_dims = np.shape(architecture['layer{}'.format(layer)][2])
        try:
            params = weight_dims[0]*weight_dims[1]*weight_dims[2]
        except:
            try:
                params = weight_dims[0]*weight_dims[1]
            except:
                try:
                    params = weight_dims[0]
                except:
                    params = 0
        total_parameters += params
    return total_parameters

def adam(g,beta_1,beta_2,m,v,t,lr):

    """Implementation of the Adam optimizer.
    This will return the running average of the mean and variance as well as the weight delta for updating parameters"""
    
    if not isinstance(g,np.ndarray):
        g = g.detach().numpy()
    if not isinstance(m,np.ndarray):
        m = m.detach().numpy()
    if not isinstance(v,np.ndarray):
        v = v.detach().numpy()
    m = beta_1 * m + (1 - beta_1) * g
    v = beta_2 * v + (1 - beta_2) * np.power(g, 2)
    m_hat = m / (1 - np.power(beta_1, t)) + (1 - beta_1) * g / (1 - np.power(beta_1, t))
    v_hat = v / (1 - np.power(beta_2, t))
    grad = lr * m_hat / (np.sqrt(v_hat + epsilon))
    grad = torch.Tensor(grad)
    grad.requires_grad = False
    m = torch.Tensor(m)
    m.requires_grad = False
    v = torch.Tensor(v)
    v.requires_grad = False
    return grad,m,v

def backward_pass(architecture,gradient_layerwise,grad_weights,grad_bias):

    """Performs a backward pass over the neural network.
    This involves calculating the gradients in each layer and storing them in the gradient_layerwise dictionary"""
    
    for layer in range(len(architecture)-1,-1,-1):
            X_input,X_output,weightsi,biasi,X_input_im2col,imi,output_shapei,kernel_shapei,stridei,operationi,imxi = architecture['layer{}'.format(layer+1)]
#             print("Operation is:{} and Layer is: {}".format(operationi,layer+1))
            if operationi == 'softmax': # Last layer -> Dont apply softmax in any layer other than the last layer!
                # not taking gradients here because we need dz_dX(secondlastlayer) which is y_pred - y
                continue
            
            if operationi == 'conv_bn_relu' or operationi == 'conv_relu' or operationi == 'conv_sigmoid' or operationi == 'conv_bn_sigmoid':
                operationi__1 = architecture['layer{}'.format(layer+2)][9]
                if operationi__1 == 'softmax':
                    y_pred = architecture['layer{}'.format(layer+2)][1]
                    y_pred = torch.reshape(y_pred,y.shape)
                    dz_dXi = y_pred - y
                    dz_dXi[dz_dXi > clip] = 0  # Gradient Clipping
                    dz_dXi[dz_dXi < -clip] = 0 # Gradient Clipping
                    X_output = torch.reshape(X_output,dz_dXi.shape)
                    if operationi == 'conv_sigmoid' or operationi == 'conv_bn_sigmoid':
                        dz_dXi *= sigmoid(X_output)*(1-sigmoid(X_output))  # Taking the derivative of the sigmoid function
                    elif 'relu' in operationi:
                        dz_dXi[X_output <= 0] = 0
                    else:
                        None
                    
                    gradient_layerwise['layer{}'.format(layer+1)][0] = dz_dXi # .
                    dz_dbi = torch.reshape(dz_dXi,biasi.shape)
                    gradient_layerwise['layer{}'.format(layer+1)][2] = dz_dbi # .
                    try:
                        dz_dweightsi = (dz_dXi).mm(torch.t(X_input_im2col))    # dz_dweightsi = dz_dXi * dXi_dweightsi  (chain rule)
                    except:
                        dz_dweightsi = (dz_dXi).mm(X_input_im2col)
                        
                    dz_dweightsi[dz_dweightsi > clip] = 0  # Gradient Clipping
                    dz_dweightsi[dz_dweightsi < -clip] = 0
                    gradient_layerwise['layer{}'.format(layer+1)][1] = dz_dweightsi #
                elif operationi__1 == 'maxpool':  # need to do something here to fix the problem
                    None

                elif 'flatten' in operationi__1:
                    # we currently have dz_doutput of flatten -> we want dz_doutput of the conv_bn_relu before flatten
                    
                    weightsi__1 = architecture['layer{}'.format(layer+2)][2] # weights2
                    dz_dXi__1 = gradient_layerwise['layer{}'.format(layer+2)][0] # dz_dXoutput of flatten
                    if len(dz_dXi__1.shape) == 3:
                        dz_dXi__1 = torch.reshape(dz_dXi__1,(-1,output_shapei__1[0]))
                    imi__1 = architecture['layer{}'.format(layer+2)][5]  # i
                    try:
                        dz_dXi = torch.t(weightsi__1).mm(dz_dXi__1)
                    except:
                        dz_dXi = weightsi__1.mm(dz_dXi__1)
                    X_output = torch.reshape(X_output,dz_dXi.shape)
                    if operationi == 'conv_sigmoid' or operationi == 'conv_bn_sigmoid':
                        dz_dXi *= sigmoid(X_output)*(1-sigmoid(X_output))    # Taking the derivative of the sigmoid function
                    elif 'relu' in operationi:
                        dz_dXi[X_output <= 0] = 0
                    else:
                        None

                    dz_dXi = torch.reshape(dz_dXi,(output_shapei[1]*output_shapei[2],-1))
                    dz_dbi = torch.reshape(dz_dXi,biasi.shape)
                    dz_dweightsi = X_input_im2col.mm(dz_dXi)
                    dz_dweightsi[dz_dweightsi > clip] = 0  # Gradient Clipping
                    dz_dweightsi[dz_dweightsi < -clip] = 0 # Gradient Clipping
                    dz_dbi = dz_dXi
                    
                    gradient_layerwise['layer{}'.format(layer+1)][0] = torch.Tensor(dz_dXi)# Can also set this to layer like in line ~800
                    
                    gradient_layerwise['layer{}'.format(layer+1)][1] = torch.Tensor(dz_dweightsi) # Can also set this to layer like in line ~800
                    
                    gradient_layerwise['layer{}'.format(layer+1)][2] = torch.Tensor(dz_dbi) # Can also set this to layer like in line ~800
                    
                else:
                    weightsi__1 = architecture['layer{}'.format(layer+2)][2]
                    dz_dXi__1 = gradient_layerwise['layer{}'.format(layer+2)][0] # dz_dX2 -> backpropagated from maxpool
                    output_shapei__1 = architecture['layer{}'.format(layer+2)][6]
                    operationi__1 == architecture['layer{}'.format(layer+2)][9] # ...
                    if len(dz_dXi__1.shape) == 3:
                        dz_dXi__1 = torch.reshape(dz_dXi__1,(-1,output_shapei__1[0]))
                    imi__1 = architecture['layer{}'.format(layer+2)][5]
                    try:
                        Y = weightsi__1.mm(dz_dXi__1)
                    except:
                        Y = weightsi__1.mm(torch.t(dz_dXi__1))
                    dz_dXi = torch.zeros(X_output.shape)
                    output_shape_current_layer = architecture['layer{}'.format(layer+1)][6]
                    bias_current_layer = architecture['layer{}'.format(layer+1)][3]
                    X_im2col_current_layer = architecture['layer{}'.format(layer+1)][4]
                    for i in range(np.shape(X_output)[0]):
                        for j in range(np.shape(X_output)[1]):
                            for k in range(np.shape(X_output)[2]):
                                idxs = getIndexes(imi__1,(i,j,k))
                                dz_dXi[i,j,k] = sum([Y[idx[0],idx[1]] for idx in idxs])
                    
                    dz_dXi[dz_dXi > clip] = 0  # Gradient Clipping
                    dz_dXi[dz_dXi < -clip] = 0 # Gradient Clipping
                    if 'sigmoid' in operationi__1: # ...
                        X_output = torch.reshape(X_output,dz_dXi.shape)
                        dz_dXi *= sigmoid(X_output)*(1-sigmoid(X_output))    # Taking the derivative of the sigmoid function
                    elif 'relu' in operationi__1: # ...
                        dz_dXi[X_output <= 0] = 0
                    else:
                        None
                    
                    dz_dXi = torch.reshape(dz_dXi,(output_shape_current_layer[1]*output_shape_current_layer[2],-1))
                    dz_dbi = torch.reshape(dz_dXi,bias_current_layer.shape)
                    dz_dweightsi = X_im2col_current_layer.mm(dz_dXi)
                    dz_dweightsi[dz_dweightsi > clip] = 0  # Gradient Clipping
                    dz_dweightsi[dz_dweightsi < -clip] = 0 # Gradient Clipping
                    gradient_layerwise['layer{}'.format(layer+1)][0] = torch.Tensor(dz_dXi)
                    gradient_layerwise['layer{}'.format(layer+1)][1] = torch.Tensor(dz_dweightsi)
                    gradient_layerwise['layer{}'.format(layer+1)][2] = torch.Tensor(dz_dbi)
                    
            if operationi == 'maxpool':
                
                weightsi__1 = architecture['layer{}'.format(layer+2)][2]
                dz_dXi__1 = gradient_layerwise['layer{}'.format(layer+2)][0] # dz_dXoutput -> backpropagated from maxpool
                output_shapei__1 = architecture['layer{}'.format(layer+2)][6]
                operationi__1 == architecture['layer{}'.format(layer+2)][9] # ...
                
                if len(dz_dXi__1.shape) == 3:
                    dz_dXi__1 = torch.reshape(dz_dXi__1,(-1,output_shapei__1[0]))
                imi__1 = architecture['layer{}'.format(layer+2)][5]
                try:
                    Y = weightsi__1.mm(dz_dXi__1)
                except:
                    try:
                        Y = weightsi__1.mm(torch.t(dz_dXi__1))
                    except:
                        Y = torch.t(weightsi__1).mm(dz_dXi__1) # Ensuring valid matrix multiplication here
                
                dz_dXi = torch.zeros(X_output.shape)
                output_shape_current_layer = architecture['layer{}'.format(layer+1)][6]
                bias_current_layer = architecture['layer{}'.format(layer+1)][3]
                X_im2col_current_layer = architecture['layer{}'.format(layer+1)][4]
                for i in range(np.shape(X_output)[0]):
                    for j in range(np.shape(X_output)[1]):
                        for k in range(np.shape(X_output)[2]):
                            idxs = getIndexes(imi__1,(i,j,k))
                            dz_dXi[i,j,k] = sum([Y[idx[0],idx[1]] for idx in idxs])

                dz_dXi[dz_dXi > clip] = 0  # Gradient Clipping
                dz_dXi[dz_dXi < -clip] = 0 # Gradient Clipping
                
                if operationi__1 == 'conv_sigmoid' or operationi__1 == 'conv_bn_sigmoid': # ...
                    X_output = torch.reshape(X_output,dz_dXi.shape)
                    dz_dXi *= sigmoid(X_output)*(1-sigmoid(X_output))    # Taking the derivative of the sigmoid function
                else:
                    dz_dXi[X_output <= 0] = 0

                gradient_layerwise['layer{}'.format(layer+1)][0] = torch.Tensor(dz_dXi)
                
                dz_dXinput = torch.zeros((X_input.shape))
                dz_dXoutput = gradient_layerwise['layer{}'.format(layer+1)][0] # output = output of maxpool

                dz_dXoutput = torch.reshape(dz_dXoutput,(output_shapei[0],X_input_im2col.shape[2]))
                
                for i in range(output_shapei[0]):
                    for j in range(X_input_im2col.shape[2]):
                        Xi2ci = X_im2col_current_layer[i,:,:]
                        idx = torch.argmax(Xi2ci[:,j]).item()
                        value = imxi[i][(idx,j)]
                        dz_dXinput[value[0],value[1],value[2]] += float(dz_dXoutput[i,j])

#                 dz_dXinput = torch.reshape(dz_dXinput,output_shapei)
                
                X_prev_im2col = architecture['layer{}'.format(layer)][4]
                X_output_prev = architecture['layer{}'.format(layer)][1]
                X_output_prev = torch.reshape(X_output_prev,dz_dXinput.shape)
                X_input_prev = architecture['layer{}'.format(layer)][0]
                prev_bias = architecture['layer{}'.format(layer)][3]
                output_shape_prev = architecture['layer{}'.format(layer)][6]
                prev_operation = architecture['layer{}'.format(layer)][9]
                
                if prev_operation == 'conv_sigmoid' or prev_operation == 'conv_bn_sigmoid':
                    dz_dXinput *= sigmoid(X_output_prev)*(1-sigmoid(X_output_prev))    # Taking the derivative of the sigmoid function
                else:
                    dz_dXinput[X_output_prev <= 0] = 0
        
                if len(dz_dXinput.shape) == 3:
                    dz_dXinput = torch.reshape(dz_dXinput,(-1,output_shape_prev[0]))
                    
                dz_dbi = torch.reshape(dz_dXinput,prev_bias.shape)
                dz_dweightsi = X_prev_im2col.mm(dz_dXinput)
                dz_dweightsi[dz_dweightsi > clip] = 0  # Gradient Clipping
                dz_dweightsi[dz_dweightsi < -clip] = 0
                
                gradient_layerwise['layer{}'.format(layer)][2] = torch.Tensor(dz_dbi)
                gradient_layerwise['layer{}'.format(layer)][1] = torch.Tensor(dz_dweightsi)
                gradient_layerwise['layer{}'.format(layer)][0] = torch.Tensor(dz_dXinput) # ...
                
            if 'flatten_dense' in operationi:
                
                operationi__1 = architecture['layer{}'.format(layer+2)][9]
                
                if operationi__1 == 'softmax':
                   
                    X_input = torch.reshape(torch.Tensor(X_input),(-1,1))
                    X_output = torch.reshape(X_output,(-1,1))
                    y_pred = architecture['layer{}'.format(layer+2)][1]
                    y_pred = torch.reshape(y_pred,y.shape)
                    dz_dXi = y_pred - y
                    dz_dXi[dz_dXi > clip] = 0  # Gradient Clipping
                    dz_dXi[dz_dXi < -clip] = 0 # Gradient Clipping
                    X_output = torch.reshape(X_output,dz_dXi.shape)
                    if 'sigmoid' in operationi:
                        dz_dXi *= sigmoid(X_output)*(1-sigmoid(X_output))  # Taking the derivative of the sigmoid function
                    elif 'relu' in operationi:
                        dz_dXi[X_output <= 0] = 0
                    else:
                        None
                    
                    dz_dbi = torch.reshape(dz_dXi,biasi.shape)
                    try:
                        dz_dweightsi = (dz_dXi).mm(torch.t(X_input))    # dz_dweightsi = dz_dXi * dXi_dweightsi (chain rule)
                    except:
                        dz_dweightsi = (dz_dXi).mm(X_input)
                        
                    dz_dweightsi[dz_dweightsi > clip] = 0  # Gradient Clipping
                    dz_dweightsi[dz_dweightsi < -clip] = 0
                    
                    gradient_layerwise['layer{}'.format(layer+1)][0] = dz_dXi # Can also set this to layer like in line ~800
                    gradient_layerwise['layer{}'.format(layer+1)][1] = dz_dweightsi # Can also set this to layer like in line ~800
                    gradient_layerwise['layer{}'.format(layer+1)][2] = dz_dbi # Can also set this to layer like in line ~800
                    
                else:
                    # Have to modify and test this before implementation -> Specifically
                    # the backprop implementation is not consistent with the ones above
                    #
                    X_output = torch.reshape(X_output,(-1,1))
                    weights__i = architecture['layer{}'.format(layer+2)][2]
                    dz_dXoutput = gradient_layerwise['layer{}'.format(layer+2)][0]
                    dz_dXoutput = torch.reshape(torch.Tensor(dz_dXoutput),X_output.shape)
                    X_input = torch.reshape(torch.Tensor(X_input),(-1,1))

                    if 'relu' in operationi:
                        dz_dXoutput[X_output<0] = 0
                        try:
                            dz_dXinput = torch.t(weights__i).mm(dz_dXoutput)
                        except:
                            dz_dXinput = torch.t(dz_dXoutput).mm(weights__i)
                        try:
                            dz_dweightsi = dz_dXoutput.mm(torch.t(X_input))
                        except:
                            dz_dweightsi = dz_dXoutput.mm(X_input)
                        dz_dbi = dz_dXoutput
                    if 'sigmoid' in operationi:
                        dz_dXoutput*= sigmoid(X_output)*(1-sigmoid(X_output))
                        try:
                            dz_dXinput = torch.t(weights__i).mm(dz_dXoutput)
                        except:
                            dz_dXinput = torch.t(dz_dXoutput).mm(weights__i)
                        try:
                            dz_dweightsi = dz_dXoutput.mm(torch.t(X_input))
                        except:
                            dz_dweightsi = dz_dXoutput.mm(X_input)
                        dz_dbi = dz_dXoutput
                    else:
                        try:
                            dz_dXinput = torch.t(weights__i).mm(dz_dXoutput)
                        except:
                            dz_dXinput = torch.t(dz_dXoutput).mm(weights__i)
                        try:
                            dz_dweightsi = dz_dXoutput.mm(torch.t(X_input))
                        except:
                            dz_dweightsi = dz_dXoutput.mm(X_input)
                        dz_dbi = dz_dXoutput
                        
                    unflattened_Xinput = architecture['layer{}'.format(layer+1)][0]
                    dz_dXinput = torch.reshape(dz_dXinput,unflattened_Xinput.shape)
                    gradient_layerwise['layer{}'.format(layer+1)][2] = torch.Tensor(dz_dbi)
                    gradient_layerwise['layer{}'.format(layer+1)][1] = torch.Tensor(dz_dweightsi)
                    gradient_layerwise['layer{}'.format(layer+1)][0] = torch.Tensor(dz_dXinput)
            
            if gradient_layerwise['layer{}'.format(layer+1)][1] is not None:
                try:
                    grad_weights['layer{}'.format(layer+1)] += gradient_layerwise['layer{}'.format(layer+1)][1]
                except:
                    grad_weights['layer{}'.format(layer+1)] += torch.t(gradient_layerwise['layer{}'.format(layer+1)][1])
            if gradient_layerwise['layer{}'.format(layer+1)][2] is not None:
                try:
                    grad_bias['layer{}'.format(layer+1)] += gradient_layerwise['layer{}'.format(layer+1)][2]
                except:
                    grad_bias['layer{}'.format(layer+1)] += torch.t(gradient_layerwise['layer{}'.format(layer+1)][2])
                    
    gc.collect()
    return

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

def zero_gradients(architecture):

    """Initialising grad weights dictionary for each layer
      for the purpose of mini-batch gradient descent"""
      
    grad_weights = {}
    grad_bias = {}
    
    # Initialising first and second moments for each gradient and bias matrix in each layer
    m = {}
    v = {}
    for layer in range(len(architecture)):
        weightsi = architecture['layer{}'.format(layer+1)][2]
        biasi = architecture['layer{}'.format(layer+1)][3]
        if weightsi is not None and biasi is not None:
            grad_weights['layer{}'.format(layer+1)] = torch.zeros(weightsi.shape)
            grad_bias['layer{}'.format(layer+1)] = torch.zeros(biasi.shape)
            m['layer{}'.format(layer+1)] = np.array([torch.zeros(weightsi.shape),torch.zeros(biasi.shape)],dtype='object')
            v['layer{}'.format(layer+1)] = np.array([torch.zeros(weightsi.shape),torch.zeros(biasi.shape)],dtype='object')
        else:
            grad_weights['layer{}'.format(layer+1)] = None
            grad_bias['layer{}'.format(layer+1)] = None
            m['layer{}'.format(layer+1)] = [None,None]
            v['layer{}'.format(layer+1)] = [None,None]
    return grad_weights,grad_bias,m,v

def update_weights(architecture,grad_weights,grad_bias,m,v,t,lr,optimizer="adam"):

    """Given the gradients, this function updates weights
    using the chosen optimizer (Adam or SGD)"""
    
    for layer in range(len(architecture)):
        if not (grad_weights['layer{}'.format(layer+1)] is None) and grad_bias['layer{}'.format(layer+1)] is not None:
            grad_weightsi = grad_weights['layer{}'.format(layer+1)]
            grad_weightsi /= bs
            grad_biasi = grad_bias['layer{}'.format(layer+1)]
            grad_biasi /= bs

            
            if optimizer.lower()=="sgd":
                # Mini-Batch SGD
                qw = lr*grad_weightsi
                qb = lr*grad_biasi
            else:
                # Mini-Batch Adam
                mw,mb = m['layer{}'.format(layer+1)]
                vw,vb = v['layer{}'.format(layer+1)]
                qw,mw,vw = adam(grad_weightsi,beta_1,beta_2,mw,vw,t,lr) # Have obtained dw
                qb,mb,vb = adam(grad_biasi,beta_1,beta_2,mb,vb,t,lr) # Have obtained db

            architecture['layer{}'.format(layer+1)][2].requires_grad = False
            architecture['layer{}'.format(layer+1)][3].requires_grad = False
            # Updating weights and biases now
            try:
                architecture['layer{}'.format(layer+1)][2] -= torch.Tensor(qw)
            except:
                architecture['layer{}'.format(layer+1)][2] -= torch.t(torch.Tensor(qw))
            try:
                architecture['layer{}'.format(layer+1)][3] -= torch.Tensor(qb)
            except:
                architecture['layer{}'.format(layer+1)][3] -= torch.t(torch.Tensor(qb))

            m['layer{}'.format(layer+1)][0] = torch.Tensor(mw)
            m['layer{}'.format(layer+1)][1] = torch.Tensor(mb)
            v['layer{}'.format(layer+1)][0] = torch.Tensor(vw)
            v['layer{}'.format(layer+1)][1] = torch.Tensor(vb)
            grad_weights['layer{}'.format(layer+1)] = torch.zeros(grad_weightsi.shape)
            grad_bias['layer{}'.format(layer+1)] = torch.zeros(grad_biasi.shape)
    return grad_weights,grad_bias,m,v


if __name__ == "__main__":
    X_inp,y_input,x_val,y_val = process_data(x_train,y_train,x_test,y_test,num_examples,val_examples)
    X_in = X_inp
    X = X_inp[0:1,:,:]
    y = torch.Tensor(y_input[:,0:1])
    
    gradient_layerwise = {} # This variable stores all the gradients in a dictionary indexed by layer (doutput_dweights and doutput_dinput)
    padding = False
    sigma = []
    architecture = {} # This variable stores all the weights and features in a dictionary indexed by layer

    #### Architecture: Can customise it to whatever you want! ####
    ##### THE NEXT FEW LINES INVOLVE BUILDING THE ARCHITECTURE OF THE CNN #####
    ####################################################################################################
    ####################################################################################################
    ####################################################################################################
    ####################################################################################################


    channels = 3
    stride = 1
    kernel_shape1 = np.array([1,3,3]) # kernel channels must match channels of input (1 here)
    X_1,weights1,X_im2col,im1,output_shape1,bias1,activation1 = conv_bn_relu(X,channels,stride,kernel_shape1,padding,batchnorm = True,activation = "relu",initialize_weights = True)
    architecture['layer1'] = np.array([X,X_1,weights1,bias1,X_im2col,im1,output_shape1,kernel_shape1,stride,'conv_bn_relu',None],dtype = 'object')
    gradient_layerwise['layer1'] = np.array([torch.zeros(X_1.shape),torch.zeros(weights1.shape),torch.zeros(bias1.shape),True],dtype='object') # Storing dz_dX, dz_dweights, dz_dbias and trainable status for each layer.
    # Trainable can be switched on and off to freeze or unfreeze the layer



    stride = 2
    kernel_shape3 = np.array([channels,3,3])
    # channels = 3 # channels in output
    X_1m,weights3,X_2_im2col,im3,output_shape3,bias3,activation3 = conv_bn_relu(X_1,channels,stride,kernel_shape3,padding,batchnorm = True,activation='relu',initialize_weights = True) # can change the number of channels to any integer value
    architecture['layer2'] = np.array([X_1,X_1m,weights3,bias3,X_2_im2col,im3,output_shape3,kernel_shape3,stride,'conv_bn_relu',None],dtype = 'object')
    gradient_layerwise['layer2'] = np.array([torch.zeros(X_1m.shape),torch.zeros(weights3.shape),torch.zeros(bias3.shape),True],dtype='object')



    # gradient layerwise format: layer i contains: dz_dXinput, dz_dXoutput, dz_dweights (Xoutput = Xinput*weights),dz_dbiasinput, dz_dbiasoutput
    kernel_shape2 = (np.array([1,3,3])) # Maxpool kernel must be 2D so that it can be applied on each channel
    stride2 = 1
    X_2,output_shape2,X_1_im2col,im2,im2_inverted = maxpool(X_1m,kernel_shape2,stride2,return_architecture=True)
    architecture['layer3'] = np.array([X_1m,X_2,None,None,X_1_im2col,im2,output_shape2,kernel_shape2,stride2,'maxpool',im2_inverted])
    gradient_layerwise['layer3'] = np.array([torch.zeros(X_2.shape),None,None,True],dtype='object')



    stride = 1
    kernel_shape3 = np.array([channels,3,3])
    X_3,weights3,X_2_im2col,im3,output_shape3,bias3,activation3 = conv_bn_relu(X_2,channels,stride,kernel_shape3,padding,batchnorm = True,activation='relu',initialize_weights = True) # can change the number of channels to any integer value
    architecture['layer4'] = np.array([X_2,X_3,weights3,bias3,X_2_im2col,im3,output_shape3,kernel_shape3,stride,'conv_bn_relu',None],dtype = 'object')
    gradient_layerwise['layer4'] = np.array([torch.zeros(X_3.shape),torch.zeros(weights3.shape),torch.zeros(bias3.shape),True],dtype='object')




    stride = 1
    kernel_shape4 = np.array([channels,3,3])
    # channels = 10 # channels in output
    channels = 10 # channels in output for next time
    X_4,weights4,X_3_im2col,im4,output_shape4,bias4,activation4 = conv_bn_relu(X_3,channels,stride,kernel_shape4,padding,activation='relu',batchnorm = True,initialize_weights = True)
    print(X_4.shape) # Debugging
    architecture['layer5'] = np.array([X_3,X_4,weights4,bias4,X_3_im2col,im4,output_shape4,kernel_shape4,stride,'conv_bn_relu',None],dtype = 'object')
    gradient_layerwise['layer5'] = np.array([torch.zeros(X_4.shape),torch.zeros(weights4.shape),torch.zeros(bias4.shape),True],dtype='object')




    stride = 2
    kernel_shape5 = np.array([channels,3,3])
    # channels = 10 # channels in output
    X_5,weights5,X_4_im2col,im5,output_shape5,bias5,activation5 = conv_bn_relu(X_4,channels,stride,kernel_shape5,padding,batchnorm = True,initialize_weights = True)
    print(X_5.shape) # Debugging
    architecture['layer6'] = np.array([X_4,X_5,weights5,bias5,X_4_im2col,im5,output_shape5,kernel_shape5,stride,'conv_relu',None],dtype = 'object')
    gradient_layerwise['layer6'] = np.array([torch.zeros(X_5.shape),torch.zeros(weights5.shape),torch.zeros(bias5.shape),True],dtype='object')




    channels = 10 # channels in output
    X_6,weights6,bias6,output_shape6 = flatten_and_dense(X_5,channels,activation = 'none', initialise_weights = True)
    architecture['layer7'] = np.array([X_5,X_6,weights6,bias6,None,None,output_shape6,channels,None,'flatten_dense_none',None],dtype = 'object')
    gradient_layerwise['layer7'] = np.array([torch.zeros(X_6.shape),torch.zeros(weights6.shape),torch.zeros(bias6.shape),True],dtype='object')
    print(X_6.shape) # Debugging
    # if layer is maxpool specify activation as maxpool



    y_pred = torch.Tensor(softmax(X_6))
    print(y_pred.shape) # Debugging
    architecture['layer8'] = np.array([X_6,y_pred,None,None,None,None,y_pred.shape,None,None,'softmax',None])
    gradient_layerwise['layer8'] = np.array([np.zeros(y_pred.shape),None,None,False],dtype='object')

    ####################################################################################################
    ####################################################################################################
    ####################################################################################################
    ####################################################################################################




    for epoch in range(epochs): # training + backpropagation. Problem: a bit slow -> need to speed it up

        gc.collect()
    #     lr = 0.0065
        lr = max(0.005,lr*np.exp(lr_multiplier*epoch)) # Reducing learning rate after each epoch
        t = 1
        loss_new_method = 0

        grad_weights,grad_bias,m,v = zero_gradients(architecture)

        for j in range(num_examples):
            sigma = []  # Collecting the sigmas from the BatchNorm layer for the purposes of gradient descent

            if j%bs == 0 or j==num_examples-1 and j != 0:
                if j%bs == 0:
                    print("Example: {}".format(j))
                    grad_weights,grad_bias,m,v = update_weights(architecture,grad_weights,grad_bias,m,v,t,lr,optimizer)
                t+= 1
            X = X_in[j:j+1,:,:]
            y = torch.Tensor(y_input[:,j:j+1])
            y_pred = forward_pass(X,architecture)
            loss = cross_entropy(y_pred.squeeze(),y.squeeze())
            loss_new_method += loss
            # Backpropagation from the last layer till the first layer
            backward_pass(architecture,gradient_layerwise,grad_weights,grad_bias)

        # Feedforward on the validation set after one epoch
        val_loss = 0
        num_correct = 0

        for index in range(val_examples):
            X = x_val[index:index+1,:,:]
            y = torch.Tensor(y_val[:,index:index+1])
            y_pred = forward_pass(X,architecture)
            vloss = cross_entropy(y_pred.squeeze(),y.squeeze())
            val_loss += vloss

            prediction = np.argmax(y_pred)
            actual = np.argmax(y)

            if prediction == actual:
                num_correct += 1
        val_accuracy = (num_correct*100)/val_examples
        val_loss /= val_examples
    #     architecture['layer5'][2] = dropout(architecture['layer5'][2])
    #     architecture['layer3'][2] = dropout(architecture['layer3'][2])
        gc.collect()

        print("Epoch: {}".format(epoch))
        print("Training Loss: {}".format(loss_new_method/num_examples))
        print("Validation Accuracy: {} %".format(val_accuracy))
        print("Validation Loss: {}".format(val_loss))

    print("Saving model weights.....")
    print("To make predictions using this model,run the function forward_pass function in the form prediction = forward_pass(input,model).....")

    with open(os.path.join(save_path),'wb') as f:
        pickle.dump(architecture,f)

