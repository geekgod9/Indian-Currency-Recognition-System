import os
import glob
from utils import *
from matplotlib import pyplot as plt
import numpy as np
#For Audio
import subprocess
import playsound
from gtts import gTTS
#To Open Dialogue box
import tkinter as tk
from tkinter import filedialog

from keras import backend as K

#Importing Open CV
import cv2 as cv

#Scikit PreProcessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle

#Import Keras Python
import keras
from keras.preprocessing import image
from keras.layers import Conv2D,Flatten, Dense, MaxPool2D,MaxPooling2D, Activation, Dropout, BatchNormalization, Input
from keras.models import Sequential, Model, load_model
from keras.utils import np_utils, to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
import time 


print("<<<<<<<< Enter the File You Want to Open >>>>>>>")
print("\n")
time.sleep(5)

root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename()

print("<<<<< You have chosen file path as :  ")
print(file_path)
time.sleep(5)

PATH = os.getcwd()
total_images = 0
data_path = PATH + '/data'
major = os.listdir(data_path)
full_path = []
image_labels = []
all_labels = ['ten','twenty']

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

time.sleep(5)

X_train = np.array(train_images, np.float32) / 255.
image_labels = to_categorical(image_labels)
print (image_labels.shape)

mean_img = X_train.mean(axis=0)
std_dev = X_train.std(axis = 0)
X_norm = (X_train - mean_img)/ std_dev
X_norm, image_labels = shuffle(X_norm, image_labels, random_state=0)

# Creating train validation split
Xtrain, Xvalid, Ytrain, Yvalid = train_test_split(X_norm, image_labels, test_size=0.2, random_state=7) 
# # load model
model = load_model('model_final.h5')
# summarize model.
print("<<<<<< This is the Model We Trained >>>>>>>> \n")
time.sleep(5)

model.summary()
score = model.evaluate(Xvalid, Yvalid, verbose=0)
print("\n")
print("Our Models accuracy is :    \n      ")
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
print("\n")
# load the model we saved
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# predicting images
print("<<<<<<<<< Here is the Output Class for your Input Image >>>>>>>>")
print("\n")
time.sleep(5)
img = cv.cvtColor(cv.imread(file_path),cv.COLOR_BGR2RGB)
img = cv.resize(img, (192,192))
img = np.reshape(img,[1,192,192,3])
img= np.array(img, np.float32) / 255.
classes = model.predict_classes(img, batch_size=1)
print ("Your predicted class is :   ",classes)

if(classes==0):
    a='Ten'
else:
    a='Twenty'
print('\nDetected denomination: Rs. ', a,"\n")
if(a=='Ten'):
    playsound.playsound('C:/Users/hp/Desktop/minor project/audio/10.mp3', True)
else:
    playsound.playsound('C:/Users/hp/Desktop/minor project/audio/20.mp3', True)

time.sleep(5)
print("<<<<<< Thank You for using our Currency Recognition System >>>>> \n")
time.sleep(5)
print("<<<<<< Created By : Mridul Goyal , Sankalp Chelani , Jayant Rana >>>>>")
