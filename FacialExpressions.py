# -*- coding: utf-8 -*-
#

# Imports
import torch
from torch.utils.data.dataset import Dataset
import urllib
import zipfile
import os
import codecs
import random
import csv
from PIL import Image


# Facial expresions
class FacialExpressions(Dataset):
    """
    Facial Expressions
    """

    # Constructor
    def __init__(self, root='./dataset', transform=None, train=True, shuffle=False):
        """
        Constructor
        :param root: Data root directory.
        :param transform: A TextTransformer object to apply.
        """
        # Properties
        self.root = root
        self.transform = transform
        self.shuffle = shuffle
        self.classes = {'neutral': 0, 'happiness': 1, 'surprise': 2, 'anger': 3, 'sadness': 4, 'disgust': 5, 'fear': 6, 'contempt': 7}

        # List of images and truth
        self.images = list()
        self.truth = dict()

        # Load file names and truth
        self._load()

        # Train
        if train:
            self.images = self.images[:12000]
        else:
            self.images = self.images[12000:]
        # end if
    # end __init__

    #############################################
    # PUBLIC
    #############################################

    #############################################
    # PRIVATE
    #############################################

    # Load dataset
    def _load(self):
        """
        Load dataset
        :return:
        """
        # Data and images path
        data_path = os.path.join(self.root, "data")
        images_path = os.path.join(self.root, "images")

        # Read CSV file
        csvfile = open(os.path.join(data_path, "legend.csv"), 'rb')

        # Load CSV
        csv_reader = csv.reader(csvfile)

        # For each row
        for row in csv_reader:
            self.truth[os.path.join(images_path, row[1])] = str(row[2].lower())
            # end for

        # For each files
        for file_name in os.listdir(images_path):
            # Image path
            image_path = os.path.join(images_path, file_name)

            # PIL image
            im = Image.open(image_path)

            # Add if size correct
            if im.size[0] == 350 and im.size[1] == 350 and image_path in self.truth:
                self.images.append(image_path)
            # end if
        # end for
    # end _load

    #############################################
    # OVERRIDE
    #############################################

    # Length
    def __len__(self):
        """
        Length
        :return:
        """
        return len(self.images)
    # end __len__

    # Get item
    def __getitem__(self, idx):
        """
        Get item
        :param idx:
        :return:
        """
        # Load image
        im = Image.open(self.images[idx])

        # Truth
        im_truth = self.truth[self.images[idx]]

        return self.transform(im), self.classes[im_truth]
    # end __getitem__

# end FacialExpressions
