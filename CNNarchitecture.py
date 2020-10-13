import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import numpy as np
# define framework
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=6,             # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (6, 28, 28)
            nn.Dropout(0.5),                # deal with overfiting
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(         # input shape (6, 14, 14)
            nn.Conv2d(6, 16, 5, 1, 0),      # output shape (16, 10, 10) no padding
            nn.Dropout(0.5),                # deal with overfiting
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (16, 5, 5)
        )
        self.conv3 = nn.Sequential(         # input shape (16, 5, 5)
            nn.Conv2d(16, 120, 5, 1, 0),    # output shape (120, 1, 1)no padding
            nn.Dropout(0.5),                # deal with overfiting
            nn.ReLU(),                      # activation
        )
        self.out1 = nn.Linear(120, 84)      # fully connected layer, output 84
        self.out2 = nn.Linear(84, 10)       # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv3 to (batch_size, 120 * 1 * 1)，
        x = self.out1(x)
        output = self.out2(x)
        return output, x                    # return x for visualization