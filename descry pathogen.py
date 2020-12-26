# -*- coding: utf-8 -*-
"""
Created on Sat May 30 18:04:54 2020

@author: ADMIN
"""

import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator 
from keras.layers.convolutional import Conv2D , MaxPooling2D
from keras.layers import  Flatten, Dropout, Dense
from keras.models import Sequential


PATH =r"E:\descry_pathogen\data_science"

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')
 
train_Pepper_healthy_dir = os.path.join(train_dir, 'Pepper_healthy')  # directory with our training pep pictures
train_Pepper_unhealthy_dir = os.path.join(train_dir, 'Pepper_unhealthy')  # directory with our training upep pictures
validation_Pepper_healthy_dir = os.path.join(validation_dir, 'Pepper_healthy')  # directory with our validation cat pictures
validation_Pepper_unhealthy_dir = os.path.join(validation_dir, 'Pepper_unhealthy')  # directory with our validation upep pictures

#### Insert images to the train folder
files_healthy = os.listdir(os.path.join(PATH, "Pepper_healthy")) # 1035, 444
files_unhealthy = os.listdir(os.path.join(PATH, "Pepper_unhealthy")) # 697 train, validation 300

total_train = len(os.listdir(train_Pepper_healthy_dir)) +len(os.listdir(train_Pepper_unhealthy_dir))
total_val = len(os.listdir(validation_Pepper_healthy_dir)) +len(os.listdir(validation_Pepper_unhealthy_dir))

 
batch_size = 135
epochs = 15  ## 8 to avoid overfitting
IMG_HEIGHT = 150
IMG_WIDTH = 150



train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', name='conv_1', 
                 input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
model.add(MaxPooling2D((2, 2), name='maxpool_1'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_2'))
model.add(MaxPooling2D((2, 2), name='maxpool_2'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_3'))
model.add(MaxPooling2D((2, 2), name='maxpool_3'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu', name='dense_1'))
model.add(Dense(128, activation='relu', name='dense_2'))
model.add(Dense(1, activation='sigmoid', name='output'))

model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)

train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

#### history.model.predict_classes()


###### Save model   
history.model.save(os.path.join(PATH, "data_science"))
from keras.models import load_model
model1 = load_model(os.path.join(PATH, "data_science"))

#### Predicting on test data
test_dir = os.path.join(PATH, 'test1')

import cv2

test_data = cv2.imread(os.path.join(test_dir, os.listdir(test_dir)[0]))
cv2.imshow('',test_data)
### preprocess
test_data1 = cv2.resize(test_data, (IMG_HEIGHT, IMG_WIDTH))

cv2.imshow('',test_data1)
test_data1 = np.expand_dims(test_data1, axis=0)

pred_test = model1.predict_classes(test_data1)
pred_test_proba = model1.predict_proba(test_data1)
pred_test =  "Pepper_unhealthy" if (pred_test.flatten()[0] == 1) else "Pepper_healthy"
pred_test_proba = int(pred_test_proba.flatten()[0])*100 

test_data = cv2.putText(test_data, "Predicted: "+pred_test+" "+str(pred_test_proba)+"%", (10,50),cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = (255,255,255), lineType = 2, thickness = 3)
cv2.imshow("", test_data)
out_name = os.listdir(test_dir)[0].split(".")[0]+"_" + pred_test + "."+ os.listdir(test_dir)[0].split(".")[1]
cv2.imwrite(os.path.join(PATH,"Prediction", out_name), test_data)
def prediction(img_name, test_dir, typ):
    test_data = cv2.imread(os.path.join(test_dir, img_name))
    ### preprocess
    test_data1 = cv2.resize(test_data, (IMG_HEIGHT, IMG_WIDTH))
    test_data1 = np.expand_dims(test_data1, axis=0)
    
    pred_test = model1.predict_classes(test_data1)
    pred_test_proba = model1.predict_proba(test_data1)
    pred_test =  "Pepper_unhealthy" if (pred_test.flatten()[0] == 1) else "Pepper_healthy"
    pred_test_proba = int(pred_test_proba.flatten()[0])*100 
    
    test_data = cv2.putText(test_data, "Predicted: "+pred_test+" "+str(pred_test_proba)+"%", (10,50),cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = (255,255,255), lineType = 2, thickness = 3)
    out_name = img_name.split(".")[0]+"_" + pred_test + "."+ img_name.split(".")[1]
    sv_path = os.path.join(PATH,"Prediction")
    cv2.imwrite(os.path.join(sv_path,typ, out_name), test_data)
    
    print("Success: "+img_name)
    return pred_test

##### Healthy Prediction
dir = os.path.join(test_dir,"healthy") 
pred_healthy = []  
for img_name in os.listdir(dir):
    
   pred= prediction(img_name, dir, typ = "healthy")
   pred_healthy.append(pred)

Accuracy_test_healthy = len([i for i in pred_healthy if i == "Pepper_healthy"])/len(pred_healthy) * 100

##### UnHealthy Prediction
dir = os.path.join(test_dir,"unhealthy") 
import re
pred_unhealthy = []  
file_names_1 = os.listdir(dir)
file_names_2 = [re.sub("\.|_+| ","", i) for i in file_names_1] 
file_names_2 = [i.split("JPG")[0]+'.JPG' for i in file_names_2]
for i in range(len(file_names_1)):
   os.rename(os.path.join(PATH, dir, file_names_1[i]), os.path.join(PATH, dir, file_names_2[i])) 
   img_name = file_names_2[i]
   pred= prediction(img_name, dir, typ = "unhealthy")
   pred_unhealthy.append(pred)

Accuracy_test_unhealthy = len([i for i in pred_unhealthy if i == "Pepper_unhealthy"])/len(pred_unhealthy) * 100


