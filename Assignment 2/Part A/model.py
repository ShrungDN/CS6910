import torch
from torch.nn import Module, Sequential
from torch.nn import Linear, Flatten, Softmax, Dropout
from torch.nn import Conv2d, MaxPool2d, AvgPool2d, AdaptiveMaxPool2d, AdaptiveAvgPool2d
from torch.nn import ReflectionPad2d, ZeroPad2d
from torch.nn import ReLU, LeakyReLU, ELU, GELU, SiLU, Mish
from torch.nn import BatchNorm2d, LayerNorm

class CNNModel(Module):
  def __init__(self, activation=ReLU, pool=MaxPool2d):
    super(CNNModel, self).__init__()
    self.act = activation
    self.pool = pool
    self.conv1 = Sequential(Conv2d(3, 64, kernel_size=5, padding=0),
                            self.act(),
                            self.pool(kernel_size=2))
    self.conv2 = Sequential(Conv2d(64, 64, kernel_size=5, padding=0),
                            self.act(),
                            self.pool(kernel_size=2))
    self.conv3 = Sequential(Conv2d(64, 64, kernel_size=5, padding=0),
                            self.act(),
                            self.pool(kernel_size=2))
    self.conv4 = Sequential(Conv2d(64, 64, kernel_size=5, padding=0),
                            self.act(),
                            self.pool(kernel_size=2))
    self.conv5 = Sequential(Conv2d(64, 64, kernel_size=5, padding=0),
                            self.act(),
                            self.pool(kernel_size=2))
    self.flatten = Flatten()
    self.fc1 = Sequential(Linear(1024, 512),
                          self.act())
    self.fc2 = Sequential(Linear(512, 10),
                          Softmax(dim=1))
        
  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.conv4(x)
    x = self.conv5(x)
    x = self.flatten(x)
    x = self.fc1(x)
    x = self.fc2(x)
    return x