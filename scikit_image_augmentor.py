#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 20:33:07 2020

@author: kemistree4
"""

import os
import random
from scipy import ndarray

# image processing library
import skimage as sk
from skimage import transform
from skimage import util
from skimage import io
from PIL import Image
from math import floor, sqrt
import numpy as np

def random_rotation(image_array: ndarray):
    # pick a random degree of rotation between 60% on the left and 60% on the right
    random_degree = random.uniform(-60, 60)
    return sk.transform.rotate(image_array, random_degree)

def random_noise(image_array: ndarray):
    # add random noise to the image
    return sk.util.random_noise(image_array)

def horizontal_flip(image_array: ndarray):
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    return image_array[:, ::-1]

#def rescale(image_array: ndarray):
    #scale_range = random.uniform(0.95, 0.60)
    img = sk.transform.rescale(image_array, scale=0.75, preserve_range=True)
    img = img.astype(np.uint8)
    return img

# dictionary of the transformations we defined earlier
available_transformations = {
    'rotate': random_rotation,
    'noise': random_noise,
    'horizontal_flip': horizontal_flip
}

folder_path = '/home/kemistree4/code/Deep-Neural-Network/bull_trout_crops'
num_files_desired = 20
destination_path = '/home/kemistree4/code/Deep-Neural-Network/bull_trout_crops/augmented'

# find all files paths from the folder
images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

num_generated_files = 0
while num_generated_files <= num_files_desired:
    # random image from the folder
    image_path = random.choice(images)
    # read image as an two dimensional array of pixels
    image_to_transform = sk.io.imread(image_path)
    # random num of transformation to apply
    num_transformations_to_apply = random.randint(1, len(available_transformations))
    num_transformations = 0
    transformed_image = None
    while num_transformations <= num_transformations_to_apply:
        if transformed_image is None:
            # random transformation to apply for a single image
            key = random.choice(list(available_transformations))
            transformed_image = available_transformations[key](image_to_transform)
            num_transformations += 1
        elif transformed_image is not None:
            # random transformation to apply for a single image
            key = random.choice(list(available_transformations))
            transformed_image = available_transformations[key](transformed_image)
            num_transformations += 1

    new_file_path = '%s/augmented_image_%s.png' % (destination_path, num_generated_files)
    
    # write image to the disk
    io.imsave(new_file_path, transformed_image)
    num_generated_files += 1