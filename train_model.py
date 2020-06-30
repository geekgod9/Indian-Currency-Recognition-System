import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
import cv2 as cv
import keras
from keras.preprocessing import image
from keras.layers import Conv2D,Flatten, Dense, MaxPool2D,MaxPooling2D, Activation, Dropout, BatchNormalization, Input
# from keras.layers.convolutional import Convolution2D
from keras.models import Sequential, Model
from keras.utils import np_utils, to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import glob

PATH = os.getcwd()
total_images = 0
data_path = PATH + '/data'
major = os.listdir(data_path)
full_path = []
image_labels = []
all_labels = ['fifty','hundred','ten','twenty']

print("Loading file structure...\n")
for a in major:
    full_path.append("data/"+a+'/')
        
        
print("Loading training images...\n")
train_images = []
for i in full_path:
    images_in_folder = 0
    label = i.split('/')[1]
    for file in glob.glob(i+"*.jpg"):
        img = cv.cvtColor(cv.imread(file),cv.COLOR_BGR2RGB)
        img = cv.resize(img, (192,192))
        total_images+=1
        train_images.append(img)
        image_labels.append(all_labels.index(label))
        images_in_folder += 1
    print("The total number of images in %s = %d" % (i,images_in_folder))
print("The total number of images in data = " + str(total_images))


X_train = np.array(train_images, np.float32) / 255.

image_labels = to_categorical(image_labels)

mean_img = X_train.mean(axis=0)
std_dev = X_train.std(axis = 0)
X_norm = (X_train - mean_img)/ std_dev
X_norm, image_labels = shuffle(X_norm, image_labels, random_state=0)

# Creating train validation split
Xtrain, Xvalid, Ytrain, Yvalid = train_test_split(X_norm, image_labels, test_size=0.2, random_state=7)

#Training the  Model
print("<<<<<< Now we will Train our model >>>>> \n")
model = Sequential()
model.add(BatchNormalization(input_shape=Xtrain.shape[1:]))
model.add(Conv2D(32, kernel_size=(3, 3), activation= 'relu',padding= 'same'))
model.add(Conv2D(32, kernel_size=(3, 3), activation= 'relu'))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Dropout(0.4))

model.add(Conv2D(64, kernel_size=(3, 3), activation= 'relu', padding='same'))
model.add(Conv2D(64, kernel_size=(3, 3), activation= 'relu'))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Dropout(0.3))

model.add(Conv2D(128, kernel_size=(3, 3), activation= 'relu', padding='same'))
model.add(Conv2D(128, kernel_size=(3, 3), activation= 'relu'))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.get_config()
model.layers[0].get_config()
model.layers[0].input_shape			
model.layers[0].output_shape			
model.layers[0].get_weights()
np.shape(model.layers[0].get_weights()[0])
model.layers[0].trainable
early_stops = EarlyStopping(patience=3, monitor='val_acc')

trained_model = model.fit(Xtrain, Ytrain, epochs = 20, shuffle = True, batch_size = 8,validation_data=(Xvalid,Yvalid))

model.save('model_final_B8.h5')
print("<<<<  Thanks for Training Your Model Has been trained >>>>>")


# # visualizing losses and accuracy
# train_loss=trained_model.history['loss']
# val_loss=trained_model.history['val_loss']
# train_acc=trained_model.history['acc']
# val_acc=trained_model.history['val_acc']
# xc=range(20)

# plt.figure(1,figsize=(7,5))
# plt.plot(xc,train_loss)
# plt.plot(xc,val_loss)
# plt.xlabel('num of Epochs')
# plt.ylabel('loss')
# plt.title('train_loss vs val_loss')
# plt.grid(True)
# plt.legend(['train','val'])
# #print plt.style.available # use bmh, classic,ggplot for big pictures
# plt.style.use(['classic'])

# plt.figure(2,figsize=(7,5))
# plt.plot(xc,train_acc)
# plt.plot(xc,val_acc)
# plt.xlabel('num of Epochs')
# plt.ylabel('accuracy')
# plt.title('train_acc vs val_acc')
# plt.grid(True)
# plt.legend(['train','val'],loc=4)
# #print plt.style.available # use bmh, classic,ggplot for big pictures
# plt.style.use(['classic'])
