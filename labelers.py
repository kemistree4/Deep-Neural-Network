#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 10 19:22:39 2020

@author: rikeem
"""

import pandas as pd
import os
from PIL import Image
from numpy import asarray
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import time


def label_image_dir(datadir='/media/kemistree4/SP PHD U3/Output frames'):
    flist = []
    
    for subdir in os.listdir(datadir):
        full_pathdir = os.path.join(datadir, subdir)
        filenames = os.listdir(full_pathdir)
        for filename in filenames:
            full_path = os.path.join(full_pathdir, filename)
            flist.append(dict(full_path=full_path, 
                              label=subdir))#, 
                              #array=asarray(Image.open(full_path))))
    return pd.DataFrame(flist)

df = label_image_dir()
random_df = df.sample(len(df))

new_df = label_image_dir(datadir='/media/kemistree4/SP PHD U3/New_Test_Set')
random_new_df = new_df.sample(len(new_df))

print(random_df)
print(random_new_df)

# Code below used to test image input
image = Image.open(random_df.loc[2438, 'full_path'])
#image.show()
# print(image.format)
# print(image.size)
# print(image.mode)

# #Converts PIL image to array to numpy array
data_array = asarray(image)

# #summarizes the shape of the array
# print(data_array.shape)

#Keras ImageDataGenerator class makes image augmentation and labeling much easier
datagen = ImageDataGenerator(rescale=1./255.)
test_datagen = ImageDataGenerator(rescale=1./255.)

#Initialize the Convolutional Neural Network
cnn_clf = Sequential()

#Add COnvolution & Pooling Layers
conv1 = Conv2D(32, (3,3), input_shape = (256, 256, 3), activation = 'relu')
cnn_clf.add(conv1)

pool1 = MaxPooling2D(pool_size = (2,2))
cnn_clf.add(pool1)

conv2 = Conv2D(32, (3,3), activation = 'relu')
cnn_clf.add(conv2)

pool2 = MaxPooling2D(pool_size = (2,2))
cnn_clf.add(pool2)

#Flatten input
cnn_clf.add(Flatten())

#Adding Dense Layers 
h1 = Dense(units = 128, activation = 'relu')
cnn_clf.add(h1)

h2 = Dense(units = 4, activation = 'sigmoid')
cnn_clf.add(h2)

#Compile the CNN
cnn_clf.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Image Augmentation
#ImageDataGenerator class generates copies of images and then performs tasks 
#like switching image orientation, shrinks its size, distorting the image to 
#create new unique images from the existing images.

datagen = ImageDataGenerator(rescale=1./255.)
test_datagen = ImageDataGenerator(rescale=1./255.)

train_generator = datagen.flow_from_dataframe(random_df[:6698],
                                              x_col='full_path',
                                              y_col='label',
                                              class_mode='categorical',
                                              batch_size=32,
                                              classes=['Bull_Trout','No_Fish', 'Kokanee', 'O_Mykiss'])

valid_generator = test_datagen.flow_from_dataframe(random_df[6698:7087],
                                              x_col='full_path',
                                              y_col='label',
                                              class_mode='categorical',
                                              batch_size=32,
                                              classes=['Bull_Trout','No_Fish', 'Kokanee', 'O_Mykiss'])

test_generator = test_datagen.flow_from_dataframe(random_df[7087:],
                                              x_col='full_path',
                                              y_col='label',
                                              shuffle=False,
                                              batch_size=32,
                                              class_mode='categorical',
                                              #class_mode=None,
                                              classes=['Bull_Trout','No_Fish', 'Kokanee', 'O_Mykiss'])

new_test_generator = test_datagen.flow_from_dataframe(random_new_df,
                                              x_col='full_path',
                                              y_col='label',
                                              shuffle=False,
                                              batch_size=32,
                                              class_mode='categorical',
                                              #class_mode=None,
                                              classes=['Bull_Trout','No_Fish', 'Kokanee', 'O_Mykiss'])

#Apply this all transformations to the training set but only the rescaling function to the test set
#train_generator = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
#test_setgen = ImageDataGenerator(rescale = .1/255)

#Find accuracy
cnn_clf.fit_generator(train_generator, steps_per_epoch = 100, epochs = 20, validation_data = valid_generator, validation_steps = (2000/32))

print("Train Set Accuracy: " + str(cnn_clf.evaluate(train_generator)))
print("Validation Set Accuracy: " + str(cnn_clf.evaluate(valid_generator)))
print("Test Set Accuracy: " + str(cnn_clf.evaluate(test_generator)))

t0 = time.time()
predictions = cnn_clf.predict(test_generator)
t1 = time.time()

print("Prediction took " + str((t1-t0)/len(predictions)) + " seconds")

t2=time.time()
predictions = cnn_clf.predict(new_test_generator)
t3 = time.time()

print("Second prediction took " + str((t3-t2)/len(predictions)) + " seconds")
print("New Test Set Accuracy: " + str(cnn_clf.evaluate(new_test_generator)))
 