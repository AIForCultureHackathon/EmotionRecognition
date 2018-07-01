#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

# Imports
import torch
import torchvision.transforms as transforms
import numpy as np
import models.EmotionClassifier
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import FacialExpressions


# Random seed
torch.manual_seed(1)
np.random.seed(1)
batch_size = 64
use_cuda = False

# Transformation to tensor and normalization
transform = transforms.Compose(
    [transforms.Resize((150, 150)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

# Download the training set
trainset = FacialExpressions.FacialExpressions(root='./dataset', train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2)

# Test set
testset = FacialExpressions.FacialExpressions(root='./dataset', train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

# Classes
classes = ('neutral', 'happiness', 'surprise', 'anger', 'sadness', 'disgust', 'fear', 'contempt')

# Dataset as iterator
dataiter = iter(trainloader)
n_batches = len(dataiter)

# Our neural net
model = models.EmotionClassifier()
if use_cuda:
    model.cuda()
# end if

# Best model
best_acc = 0.0

# Objective function is cross-entropy
criterion = nn.CrossEntropyLoss()

# Learning rate
learning_rate = 0.001

# Stochastic Gradient Descent
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# Nb iterations
n_iterations = 60
bootstrap = 10
n_fail = 0

# Training !
for epoch in range(10000):
    # Average loss during training
    average_train_loss = 0.0
    average_test_loss = 0.0

    # Iterate over batches
    for i, data in enumerate(trainloader, 0):
        # Get the inputs and labels
        inputs, labels = data

        # To variable
        if use_cuda:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)
        # end if

        # Put grad to zero
        optimizer.zero_grad()

        # Forward
        outputs = model(inputs)

        # Loss
        loss = criterion(outputs, labels)

        # Backward
        loss.backward()

        # Optimize
        optimizer.step()

        # Add to loss
        average_train_loss += loss.data[0]

        # Take the max as predicted
        _, predicted = torch.max(outputs.data, 1)
    # end for

    # Test model on test set
    success = 0
    total = 0
    for i, data in enumerate(trainloader, 0):
        # Get the inputs and labels
        inputs, labels = data

        # To variable
        if use_cuda:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)
        # end if

        # Neural net's output
        outputs = model(inputs)

        # Loss
        loss = criterion(outputs, labels)

        # Take the max is predicted
        _, predicted = torch.max(outputs.data, 1)

        # Add to total
        total += labels.size(0)

        # Add to loss
        average_test_loss += loss.data[0]

        # Add correctly classified images
        success += (predicted == labels.data).sum()
    # end for

    # Print average loss
    print(u"Epoch {}, average loss {}, test loss {}, test accuracy {}".format(
        epoch,
        average_train_loss / n_batches,
        average_test_loss / n_batches,
        100.0 * success / total
        )
    )

    # Save
    if success / total * 100.0 > best_acc and epoch > bootstrap:
        print(u"Saving model...")
        best_acc = success / total * 100.0
        torch.save(model.state_dict(), open('model.pth', 'wb'))
        n_fail = 0
    elif epoch > bootstrap:
        n_fail += 1
    # end if

    # Check
    if n_fail > n_iterations:
        break
    # end if
# end for


