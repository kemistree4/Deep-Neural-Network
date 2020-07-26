#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Baseline MLP for MNIST dataset
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)
# flatten 28*28 images to a 784 vector for each image
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape((X_train.shape[0], num_pixels)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], num_pixels)).astype('float32')
X_val = X_val.reshape((X_val.shape[0], num_pixels)).astype('float32')
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
X_val = X_val / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
y_val = np_utils.to_categorical(y_val)
num_classes = y_train.shape[1]

# define baseline model
def baseline_model(num_neurons=[500,50]):
	# create model
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
    #model.add(Dropout(.8))
    for i in range(len(num_neurons)):
        model.add(Dropout(.45))
        model.add(Dense(num_neurons[i], kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.452))
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# build the model
model = baseline_model()

# Fit the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=200)

# Final evaluation of the model
scores = model.evaluate(X_val, y_val, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

print("Train Set Accuracy: " + str(model.evaluate(X_train, y_train)[1]*100))
print("Validation Set Accuracy: " + str(model.evaluate(X_val, y_val)[1]*100))
print("Test Set Accuracy: " + str(model.evaluate(X_test, y_test)[1]*100))


