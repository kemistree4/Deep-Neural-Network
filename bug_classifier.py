#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 19:35:02 2020

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
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator
import time
from keras.applications import resnet50

classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20'
           , '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', 
           '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59',
            '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78',
             '78', '79', '80', '81', '82', '83', '84', '85', '86', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95',
              '96', '97', '98', '99', '100', '101']

def label_image_dir(datadir='/media/kemistree4/SP PHD U3/ip102_v1.1/images'):
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

print(random_df)

# Code below used to test image input
image = Image.open(random_df.loc[2438, 'full_path'])
#image.show()
# print(image.format)
# print(image.size)
# print(image.mode)

# #Converts PIL image to array to numpy array
data_array = asarray(image)

cnn_clf = resnet50.ResNet50(include_top=True, weights='imagenet')
cnn_clf.load_weights('resnet50_weights_tf_dim_ordering_tf_kernels.h5')
cnn_clf.compile(optimizer='rmsprop', loss='categorical_crossentropy')

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
h1 = Dense(units = 250, activation = 'relu')
cnn_clf.add(h1)

d1 = Dropout(0.4)
cnn_clf.add(d1)

h2 = Dense(units = 102, activation = 'sigmoid')
cnn_clf.add(h2)

#Compile the CNN
cnn_clf.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


#Image Augmentation
#ImageDataGenerator class generates copies of images and then performs tasks 
#like switching image orientation, shrinks its size, distorting the image to 
#create new unique images from the existing images.

datagen = ImageDataGenerator(rescale=1./255.)
test_datagen = ImageDataGenerator(rescale=1./255.)

train_generator = datagen.flow_from_dataframe(random_df[:26212],
                                              x_col='full_path',
                                              y_col='label',
                                              class_mode='categorical',
                                              batch_size=32,
                                              classes=classes)

valid_generator = test_datagen.flow_from_dataframe(random_df[26212:29489],
                                              x_col='full_path',
                                              y_col='label',
                                              class_mode='categorical',
                                              batch_size=32,
                                              classes=classes)

test_generator = test_datagen.flow_from_dataframe(random_df[29489:],
                                              x_col='full_path',
                                              y_col='label',
                                              shuffle=False,
                                              batch_size=32,
                                              class_mode='categorical',
                                              classes=classes)



#Apply this all transformations to the training set but only the rescaling function to the test set
#train_generator = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
#test_setgen = ImageDataGenerator(rescale = .1/255)

#Find accuracy
cnn_clf.fit(train_generator, steps_per_epoch = 50, epochs = 30, validation_data = valid_generator, validation_steps = (2000/32))

print("Train Set Accuracy: " + str(cnn_clf.evaluate(train_generator)))
print("Validation Set Accuracy: " + str(cnn_clf.evaluate(valid_generator)))
print("Test Set Accuracy: " + str(cnn_clf.evaluate(test_generator)))

t0 = time.time()
predictions = cnn_clf.predict_generator(test_generator)
t1 = time.time()

print("Prediction took " + str((t1-t0)/len(predictions)) + " seconds")


