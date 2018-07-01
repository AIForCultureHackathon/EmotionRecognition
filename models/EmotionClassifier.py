#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


# Neural net
class EmotionClassifier(nn.Module):
    """
    Neural net
    """

    # Constructor
    def __init__(self):
        """
        Constructor
        """
        super(EmotionClassifier, self).__init__()
        self.conv_layer1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv_layer2 = nn.Conv2d(6, 16, 5)
        self.linear_layer1 = nn.Linear(16 * 34 * 34, 300)
        self.linear_layer2 = nn.Linear(300, 8)
    # end __init__

    # Forward pass
    def forward(self, x):
        """
        Forward pass
        :param x:
        :return:
        """
        # Conv layer
        x = self.conv_layer1(x)
        x = F.relu(x)

        # Max pooling
        x = self.pool(x)

        # Conv layer 2
        x = self.conv_layer2(x)
        x = F.relu(x)

        # Max pooling 2
        x = self.pool(x)

        x = x.view(-1, 16 * 34 * 34)

        # 2 linear layers
        x = F.relu(self.linear_layer1(x))
        x = F.relu(self.linear_layer2(x))

        return F.log_softmax(x)
    # end forward

# end EmotionClassifier
