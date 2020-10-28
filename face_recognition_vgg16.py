#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 18:09:17 2020

@author: siddharthsmac
"""

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from glob import glob
import tensorflow as tf

IMAGE_SIZE = [224, 224]
train_path = '/users/siddharthsmac/desktop/Mypics/Train'
test_path = '/users/siddharthsmac/desktop/Mypics/Test'

vgg = VGG16(input_shape = IMAGE_SIZE + [3], weights = 'imagenet', include_top = False)

for layer in vgg.layers:
  layer.trainable = False

folders = glob('/users/siddharthsmac/desktop/Mypics/Train/*')

x = Flatten()(vgg.output)
prediction = Dense(len(folders), activation = 'softmax')(x)

model = Model(inputs = vgg.input, outputs = prediction)
model.summary()

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('/users/siddharthsmac/desktop/Mypics/train', target_size = (224, 224), batch_size = 32, class_mode = 'categorical')
test_set = test_datagen.flow_from_directory('/users/siddharthsmac/desktop/Mypics/test', target_size = (224, 224), batch_size = 32, class_mode = 'categorical')

r = model.fit_generator(training_set, validation_data = test_set, epochs = 20, steps_per_epoch = len(training_set), validation_steps = len(test_set))

import matplotlib.pyplot as plt

plt.plot(r.history['loss'], label = 'train loss')
plt.plot(r.history['val_loss'], label = 'val loss')
plt.legend()
plt.show()

plt.plot(r.history['acc'], label = 'train acc')
plt.plot(r.history['val_acc'], label = 'val acc')
plt.legend()
plt.show()

model.save('facefeatures_new_model.h5')
