## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale             image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for             each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but         don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 4x4           square convolution kernel
        ## output size = (W-F)/S +1 = (224-5)/1 +1 = 220
        self.conv1 = nn.Conv2d(1, 32, 5)
        # the output Tensor for one image, will have the dimensions: (32,220,220)
        # after one pool layer, this becomes (32, 110, 110)
        
        #applying batchnorm
        self.bn1_1 = nn.BatchNorm2d(32)
        
        # second conv layer: 32 inputs, 64 outputs, 3x3 conv
        ## output size = (W-F)/S +1 = (110-3)/1 +1 = 108
        self.conv2 = nn.Conv2d(32, 64, 3)
        # the output tensor will have dimensions: (64, 108, 108)
        # after one pool layer, this becomes (64, 54, 54)
        
        #applying batchnorm
        self.bn1_2 = nn.BatchNorm2d(64)
        
        # third conv layer: 64 inputs, 128 outputs, 3x3 conv
        ## output size = (W-F)/S +1 = (54-3)/1 +1 = 52
        self.conv3 = nn.Conv2d(64, 128, 3)
        # the output tensor will have dimensions: (128, 52, 52)
        # after one pool layer, this becomes (128, 26, 26)
        
        #applying batchnorm
        self.bn1_3 = nn.BatchNorm2d(128)
        
        #fourth conv layer: 128 inputs, 256 outputs, 3x3 conv
        ## output size = (W-F)/S +1 = (26-3)/1 +1 = 24
        self.conv4 = nn.Conv2d(128, 256, 3)
        # the output tensor will have dimensions: (256, 24, 24)
        # after one pool layer, this becomes (256, 12, 12)
        
        #applying batchnorm
        self.bn1_4 = nn.BatchNorm2d(256)
        
         #fifth conv layer: 256 inputs, 512 outputs, 1x1 conv
        ## output size = (W-F)/S +1 = (12-3)/1 +1 = 10
        self.conv5 = nn.Conv2d(256, 512, 3)
        # the output tensor will have dimensions: (512, 10, 10)
        # after one pool layer, this becomes (512, 5, 5)
        
        # pool with kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2, 2)
        
        
        # 1024 outputs * 5*5 filtered/pooled map size
        self.fc1 = nn.Linear(512*5*5, 1024)
        
        # also consider adding a dropout layer to avoid overfitting
        
        
        self.fc1_drop = nn.Dropout(p=0.4)
        
        self.fc2 = nn.Linear(1024, 136)
       
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other         layers (such as dropout or batch normalization) to avoid overfitting
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        # first activated conv layer
        x = F.relu(self.conv1(x))
        
        
        # applies pooling layer
        x = self.pool(x)
        
        #applies batchnorm layer
        x = self.bn1_1(x)
        
        # second activated conv layer
        x = F.relu(self.conv2(x))
        
        
        # applies pooling layer
        x = self.pool(x)
        
        #applies batchnorm layer
        x = self.bn1_2(x)
        
        # third activated conv layer
        x = F.relu(self.conv3(x))
        
        
        # applies pooling layer
        x = self.pool(x)
        
        #applies batchnorm layer
        x = self.bn1_3(x)
        
        # fourth activated conv layer
        x = F.relu(self.conv4(x))
        
        
        # applies pooling layer
        x = self.pool(x)
        
        #applies batchnorm layer
        x = self.bn1_4(x)
        
        # fifth activated conv layer
        x = F.relu(self.conv5(x))
        
        # applies pooling layer
        x = self.pool(x)
        
        # prep for linear layer
        # this line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)
        
        # two linear layers with dropout in between
        x = self.fc1(x)
        
        x = self.fc1_drop(x)
        
        
        x = self.fc2(x)
        
       
       
        # a modified x, having gone through all the layers of your model, should be returned
        return x
