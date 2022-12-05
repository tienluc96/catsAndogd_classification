# Del Daix 2019
# Simple TensorFlow binary classifier with step by step vizualisation of convoluted images
# based on colab.research.google.com
# dataset found using: !wget -c https://s3.amazonaws.com/content.udacity-data.com/nd089/Cat_Dog_data.zip


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers, Model, optimizers, regularizers
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import random
from pathlib import Path
from tensorflow.keras.optimizers import RMSprop, SGD, Adam, Adamax

dataset_dir = 'E:\etude\Semestre 2\ML\cats_and_dogs_filtered\cats_and_dogs_filtered'
training_dir = os.path.join(dataset_dir, 'train')
val_dir = os.path.join(dataset_dir, 'validation')

#directory with our training cat images
train_cats_dir = os.path.join(training_dir,'cats')

#directory with our training dog images
train_dogs_dir = os.path.join(training_dir,'dogs')

#directory with our validation cat images
validation_cats_dir = os.path.join(val_dir,'cats')

#directory with our validation dog images
validation_dogs_dir = os.path.join(val_dir,'dogs')




# building the convolutional neural network

# we have 2D color images of dimension 150x150

image_in = layers.Input(shape=(150,150,3))


# # next option: complexify the model with added hidden layers
x = layers.Conv2D(32,(3,3),activation='relu')(image_in)
x = layers.MaxPooling2D(pool_size=(2,2))(x)
# x = layers.BatchNormalization()(x)
# x = layers.Dropout(0.5)(x)

x = layers.Conv2D(64,(3,3),activation='relu')(x)
x = layers.MaxPooling2D(pool_size=(2,2))(x)
# x = layers.BatchNormalization()(x)
# x = layers.Dropout(0.5)(x)

x = layers.Conv2D(128,(3,3),activation='relu')(x)
x = layers.MaxPooling2D(pool_size=(2,2))(x)
# x = layers.BatchNormalization()(x)
# x = layers.Dropout(0.5)(x)

x = layers.Conv2D(256,(3,3),activation='relu')(x)
x = layers.MaxPooling2D(pool_size=(2,2))(x)
# x = layers.BatchNormalization()(x)
# x = layers.Dropout(0.5)(x)

# fully connected layer with 512 hidden units

x = layers.Flatten()(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.7)(x)
# output layer of 1 for cat or dog binary result

output = layers.Dense(1,activation='sigmoid')(x)

my_model = Model(image_in, output)
my_model.summary()

# model configuration: optimize with GradientDescentOptimizer, RMSProp or AdamOptimizer for example

my_model.compile(loss='binary_crossentropy',optimizer=Adam(lr=0.0001), metrics=['accuracy'])





#all images will be rescaled and modifies the input
train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=45,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.5,
                                    horizontal_flip=True
                                    )
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
# prepare the datasets


train_dataset = train_datagen.flow_from_directory(training_dir, 
                                                  target_size=(150,150), 
                                                  batch_size=20, 
                                                  class_mode='binary')

val_dataset = val_datagen.flow_from_directory(val_dir,
                                          target_size=(150,150), 
                                          batch_size=20, 
                                          class_mode='binary')

# train the model on 2,000 images over 15 epochs, validate on 1,000 images

history = my_model.fit_generator(train_dataset,
                                 steps_per_epoch=100, 
                                 epochs=15, 
                                 validation_data=val_dataset, 
                                 validation_steps=50, 
                                 verbose=2)
# save model
my_model.save('E:/etude/Semestre 2/ML/cats_and_dogs_filtered/cats_and_dogs_filtered/trainingmodel/my_model.h5')

#display the accuracy and the loss per epoch
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))


plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
a,=plt.plot(epochs, acc)
b,=plt.plot(epochs, val_acc)
plt.title('Training and validation accuracy')
plt.legend([a,b],['train accuracy','val accuracy'],loc='lower right')

# Plot training and validation loss per epoch
plt.subplot(1,2,2)
c,=plt.plot(epochs, loss)
d,=plt.plot( epochs,val_loss)
plt.legend([c,d],['train loss','val loss'],loc='upper left')
plt.title('Training and validation loss')

plt.show()

import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
# Predict the values from the validation dataset
import cv2

#load the model
new_model = tf.keras.models.load_model('E:/etude/Semestre 2/ML/Cat_Dog_data/my_model.h5')

# calculate the predict
y_pred = new_model.predict_generator(val_dataset)
y_p = np.where(y_pred > 0.5, 1,0)
confusion_mtx = confusion_matrix(val_dataset.classes, y_p) 
# plot the confusion matrix
f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Blues",linecolor="gray", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Generate a classification report
report = classification_report(val_dataset.classes, y_p, target_names=['0','1'])
print(report)




