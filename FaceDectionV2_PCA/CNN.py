from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense

import cv2
import scipy.linalg as s_linalg
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from keras.models import Sequential, Model, load_model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, Activation
from keras.optimizerS import Adam, SGD, RMSprop
from keras.callbacks import Callback, EarlyStopping
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt

from PCA import pca_class
from TwoDPCA import twoDPcaClass
from TwoD_Square_PCA import twoDSquarePcaClass

from imageMatrix import imageToMatrixClass
from images_matrix_for_2d_square_pca import imagesToMatrixClassForTwoD
from dataset import datasetClass


# no_of_images_of_one_person = 8
# dataset_obj = datasetClass(no_of_images_of_one_person)
# #Data for training
# imgNamesTraining = dataset_obj.imgPathTraining
# labelsTraining = dataset_obj.labelsTraining
# imgWidth, imgHeight = 50, 50
# imageToMatrixClassObj = imageToMatrixClass(imgNamesTraining, imgWidth, imgHeight)
# imgMatrix = imageToMatrixClassObj.get_matrix()

no_of_images_of_one_person = 8
dataset_obj = datasetClass(no_of_images_of_one_person)
#Data for training
imgNamesTraining = dataset_obj.imgPathTraining
labelsTraining = dataset_obj.labelsTraining
NumImagesTraining = dataset_obj.NumImagesTraining
# imagesTarget = dataset_obj.imagesTargetArray
imagesTargetArray = dataset_obj.imagesTargetArray

#Data for Testing
imgPathTesting = dataset_obj.imgPathTesting
labelsTesting = dataset_obj.labelsTesting
NumImagesTesting = dataset_obj.NumImagesTesting

# data_dir = ("images/ORL")
print(imgNamesTraining.shape)
print(imgPathTesting.shape)

df=pd.DataFrame(imgNamesTraining)
df.columns=['images']
df['labels']=labelsTraining
df=df.sample(frac=1).reset_index(drop=True)

train_data = df



def plot(image_batch, label_batch):
    plt.figure(figsize=(10,5))
    for i in range(10):
        ax = plt.subplot(2,5,i+1)
        img = cv2.imread(str(image_batch[i]))
        img = cv2.resize(img, (224,224))
        plt.imshow(img)
        plt.tilte(label_batch[i])
        plt.axis("off")

def prepare_and_load(isval=true):
    


#     def __init__(self, dir, y, targetNames, NumOfElements, QualityPercent):
#         self.dir = dir




# X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
# X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],X_test.shape[2],1))

# X_train=X_train/255
# X_test=X_test/255

model = Sequential()
model.add(Conv2D(32,(3,3),acivation='relu',input_shape=(244, 224,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten()) # Converts 3D feature maps to 1D feature vectors

model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(2))
model.add(Activation('softmax'))

batch_size = 16
nb_epochs = 3

# Get a train Data generator
train_data = data_gen(data=train_data, batch_size=batch_size)

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit_generator(train_data_gen, epochs=nb_epochs, steps_per_epoch=nb_train_steps, validation_data=(val_data, val_labels))

model.evaluate(X_test,y_test)

# Image Classifications model
def vgg16(num_classes=None):
    model = VGG16(weights='imagenet', include_top=True, input_shape())