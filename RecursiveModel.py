import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
import sys
sys.setrecursionlimit(30000)



multiply = lambda x: x[0]*multiply(x[1:]) if len(x) > 1 else x[0]

class Function(Enum):
    RESIZE = 1

class Recursive_model(nn.Module):
    def __init__(self, input_size, maximum_recursion_time=5):
        super().__init__()
        self.activation_function = F.relu
        self.maximum_recursion_time = maximum_recursion_time

        self.layer_list = []
        self.input_size = input_size
        self.output_size = input_size

        self.ConvolutionalModel = None

        self.Block_layer_modification = False

    def set_ConvolutionalModel(self, ConvolutionalModel): #NECESSARY
        self.ConvolutionalModel = ConvolutionalModel

    def add_convolutional_layer(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, *para, **kparam):
        if self.Block_layer_modification:
            raise NotImplementedError
        self.layer_list.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, *para, **kparam))
        self.output_size = (out_channels, (self.output_size[1]-kernel_size+2*padding)//stride +1, (self.output_size[2]-kernel_size+2*padding)//stride +1)

    def add_linear_layer(self, hidden_layer, *para, **kparam):
        if self.Block_layer_modification:
            raise NotImplementedError
        if not isinstance(self.output_size,int):
            self.add_resize_layer((multiply(self.output_size),))
        self.layer_list.append(nn.Linear(multiply(self.output_size), hidden_layer, *para, **kparam))
        self.output_size = hidden_layer

    def add_resize_layer(self, new_size):
        if self.Block_layer_modification:
            raise NotImplementedError
        self.output_size = new_size
        self.layer_list.append(lambda x : x.reshape(-1,*new_size))

    def set_activation_function(self, activation_function):
        self.activation_function = activation_function

    @property
    def shape(self):
        return self.output_size

    def forward(self, x,current_recursion_floor=0):
        if self.ConvolutionalModel is None:
            raise NotImplementedError

        for layer in self.layer_list:
            x = layer(x)
            x = self.activation_function(x)

        if current_recursion_floor < self.maximum_recursion_time:
            x = self.ConvolutionalModel(x, current_recursion_floor)
        return x
