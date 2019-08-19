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

#Scaling the Data
from sklearn.preprocessing import StandardScaler

feature_scaler = StandardScaler()

feature = feature_scaler.fit_transform(features)

#Importing Keras and Subsequent Libraries
import keras
from keras.models import Sequential
from keras.layers import Dense

def train_classifier():
    ann_clf = Sequential() #makes sequential callable?

    #Adding Input and Hidden Layers
    h1 = Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11) 
    #units = number of output nodes in the layer 
    #kernel_initializer = type of weights that should be used initially for the layer
    #activation = which activation function to use
    ann_clf.add(h1)
    h2 = Dense(units = 9, kernel_initializer = 'uniform', activation = 'relu')
    ann_clf.add(h2)
    
    op = Dense(units = 6, kernel_initializer = 'uniform', activation = 'softmax')
    ann_clf.add(op)
    
    #Compiling
    ann_clf.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return ann_clf

#Implementing Cross-Validation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

#Training the Neural Network
keras_ann_clf = KerasClassifier(build_fn = train_classifier, batch_size = 10, epochs = 500)

results = cross_val_score(estimator = keras_ann_clf, X = features, y = labels, cv = 10)

print(results)
print(results.std())
