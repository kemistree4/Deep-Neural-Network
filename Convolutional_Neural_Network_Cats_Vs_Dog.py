#Import libraries
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Initialize the Convolutional Neural Network
cnn_clf = Sequential()

#Add COnvolution & Pooling Layers
conv1 = Conv2D(32, (3,3), input_shape = (64, 64, 3), activation = 'relu')
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

h2 = Dense(units = 1, activation = 'sigmoid')
cnn_clf.add(h2)

#Compile the CNN
cnn_clf.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Image Augmentation
#ImageDataGenerator class generates copies of images and then performs tasks 
#like switching image orientation, shrinks its size, distorting the image to 
#create new unique images from the existing images.

from keras.preprocessing.image import ImageDataGenerator

#Apply this all transformations to the training set but only the rescaling function to the test set
train_setgen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_setgen = ImageDataGenerator(rescale = .1/255)

training_data = train_setgen.flow_from_directory(r'C:\Users\Rikeem\Desktop\Datasets\animal_data\training_set', target_size = (64, 64), batch_size = 32, class_mode = 'binary')
test_data = train_setgen.flow_from_directory(r'C:\Users\Rikeem\Desktop\Datasets\animal_data\test_set', target_size = (64, 64), batch_size = 32, class_mode = 'binary')
#flow_from_directory = path of directory where images are stored
#target_size = says the images are 64 by 64 pixels
#batch_size = batch size of images that you want to generate?
#class_mode = binary because we have two classes as output, cat or dog

#Find accuracy
cnn_clf.fit_generator(training_data, steps_per_epoch = (8000/32), epochs = 25, validation_data = test_data, validation_steps = (2000/32))
