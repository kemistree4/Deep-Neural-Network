#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 21:04:13 2020

@author: rikeem
"""
import numpy as np
from numpy import asarray
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

def fish_predict(model = keras.models.load_model('/media/rikeem/SP PHD U3/fish_model/'), image_path='/home/rikeem/Desktop/Test_set_2125.png'):
    input_image = tf.keras.preprocessing.image.load_img(path=image_path, 
                                                  grayscale=False, 
                                                  color_mode='rgb', 
                                                  target_size=(256, 256))
    image_array = asarray(input_image)
    rimage = image_array.reshape(1,256,256,3)   
    predictions = model.predict(x=rimage, verbose=0)
    results = predictions
    
    #Find highest element in results numpy array
    maxElement = np.amax(results)
    max_index = np.where(results == maxElement)
    
    plt.imshow(input_image)
    if maxElement == 0: 
        print("I have no idea...sorry")
    elif max_index [1][0] == 0:
        print("I think this is a Bull Trout")
    elif max_index [1][0] == 1:
        print("I think this is a Kokanee")
    elif max_index [1][0] == 2:
        print("I don't see a fish here")
    elif max_index [1][0] == 3:
        print("I think this is a juvenile O. mykiss")
    return predictions

                                    