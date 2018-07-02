#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

# Imports
from __future__ import print_function, division
import os
import torch
import pandas as pd
from FaceLandmarksDataset import FaceLandmarksDataset
from skimage import io, transform
import numpy as np
from Transforms import Rescale, RandomCrop
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import warnings
warnings.filterwarnings("ignore")

landmarks_frame = pd.read_csv('faces/face_landmarks.csv')

n = 65
img_name = landmarks_frame.iloc[n, 0]
landmarks = landmarks_frame.iloc[n, 1:].as_matrix()
landmarks = landmarks.astype('float').reshape(-1, 2)

print('Image name: {}'.format(img_name))
print('Landmarks shape: {}'.format(landmarks.shape))
print('First 4 Landmarks: {}'.format(landmarks[:4]))

scale = Rescale(256)
crop = RandomCrop(128)
composed = transforms.Compose([
    Rescale(256),
    RandomCrop(224),
    transforms.ToTensor()
])

transformed_dataset = FaceLandmarksDataset(csv_file='faces/face_landmarks.csv',
                                           root_dir='faces/',
                                           transform=composed)

dataloader = DataLoader(transformed_dataset, batch_size=4,
                        shuffle=True, num_workers=4)

for i in range(len(transformed_dataset)):
    sample = transformed_dataset[i]

    print(i, sample['image'].size(), sample['landmarks'].size())

    if i == 3:
        break

for i_batch, sample_batched in enumerate(dataloader):
