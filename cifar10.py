#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: kemistree4
"""

import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# Dataset is 60000 images with 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))


model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.4))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.4))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(10))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=40, 
                    validation_data=(test_images, test_labels))

# Final evaluation of the model

print("Train Set Accuracy: " + str(model.evaluate(train_images, train_labels)[1]*100))
print("Test Set Accuracy: " + str(model.evaluate(test_images, test_labels)[1]*100))