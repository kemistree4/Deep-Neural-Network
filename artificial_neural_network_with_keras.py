#Import libraries
import pandas as pd 
import numpy as np

#Import dataset
redwine_data = pd.read_csv(r'C:\Users\Rikeem\Desktop\Datasets\redwine_data.csv', sep=';')

#Grouping dataset. First 11 columns are properties of wine, last column is quality rating of wine
features = redwine_data.iloc[:,0:11].values
labels = redwine_data.iloc[:,11].values

#One Hot Encoding the Output
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder() #Label Encoder class converts labels into a series of integers. 

y = encoder.fit_transform(labels)

labels = pd.get_dummies(y).values #get_dummies converts integers into one hot encoded matrix

#Splits the data into trainingn and test sets

from sklearn.model_selection import train_test_split

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.2, random_state = 0)

#Scaling the Data
from sklearn.preprocessing import StandardScaler

feature_scaler = StandardScaler()
train_features = feature_scaler.fit_transform(train_features)
test_features = feature_scaler.transform(test_features)

#Importing Keras and Subsequent Libraries
import keras
from keras.models import Sequential
from keras.layers import Dense

ann_clf = Sequential() #makes sequential callable?

#Adding Input and Hidden Layers
h1 = Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11) 
#units = number of output nodes in the layer 
#kernel_initializer = type of weights that should be used initially for the layer
#activation = which activation function to use
ann_clf.add(h1)
h2 = Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu')
ann_clf.add(h2)

#Creating the output layer
op = Dense(units = 6, kernel_initializer = 'uniform', activation = 'softmax')
ann_clf.add(op)

#Compiling
ann_clf.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Training the Neural Network
ann_clf.fit(train_features, train_labels, batch_size=1, epochs=500)

#Making predictions. Predictions are basically just the test to see 
#how well your network properly identifies an unlabeled sample
predictions = ann_clf.predict(test_features)
predictions = np.argmax(predictions, axis=1)
test_labels = np.argmax(test_labels, axis=1)

#Constructing confusion matrix and calculate accuracy
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(test_labels, predictions))
print(classification_report(test_labels, predictions))
print(accuracy_score(test_labels, predictions))
