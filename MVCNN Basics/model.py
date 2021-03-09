import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_classes):
        
        """  
        DEFINE YOUR NETWORK HERE 
        """
        
        super(CNN, self).__init__()
        # Convolution Layer 1 with Leaky ReLu
        self.num_classes = num_classes
        self.conv1  = nn.Conv2d(in_channels= 1, out_channels=16, kernel_size=8, stride=2, padding=0, bias=True)
        self.lrelu1 = nn.LeakyReLU(0.1)
        # MaxPooling Layer 
        self.pool1  = nn.MaxPool2d(kernel_size=2,stride=2)
        # Convolution Layer 2 with Leaky ReLu
        self.conv2  = nn.Conv2d(in_channels= 16, out_channels=32, kernel_size=4, stride=2, padding=0, bias=True)
        self.lrelu2 = nn.LeakyReLU(0.1)
        # Maxpooling Layer 
        self.pool2  = nn.MaxPool2d(kernel_size=2,stride=2)
        # Implementing fully convolutional layer as fully connected layer
        self.conv3  = nn.Conv2d(in_channels=32,out_channels=10,kernel_size=6,padding = 0)
        
    def forward(self, x):
        """

        DEFINE YOUR FORWARD PASS HERE

        """
        
        out = self.conv1(x)
        out = self.lrelu1(out)
        out = self.pool1(out)
        
        out = self.conv2(out)
        out = self.lrelu2(out)
        out = self.pool2(out)
        
        out = self.conv3(out)
        
        return out