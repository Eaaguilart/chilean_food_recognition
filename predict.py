import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

sns.set(style='darkgrid', palette='cubehelix')

from tensorflow.keras.models import Sequential
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import regularizers, optimizers
from tensorflow.python.client import device_lib
from scheduler import *


from glob import glob

from tensorflow.keras.applications import EfficientNetB0

conv_base = EfficientNetB0(include_top = False,
                      weights = "imagenet",
                      input_tensor = None,
                      input_shape = (224, 224, 3),
                      classes = 2)

model = Sequential()
model.add(conv_base)
model.add(tf.keras.layers.GlobalMaxPooling2D(name='max_pool'))
model.add(Dropout(0.2, name='top_dropout'))
model.add(Dense(2, activation='softmax', name='fc_out'))

model.load_weights("foodnonfood.h5")

root = "/home/eduardo-ucn/Documents/datasets/ChileanFood64/images/"

for folder in glob(root+"*"):
    print(folder)
    if not "Leche A" in folder:
        continue
    for img_path in glob(folder+"/*.*"):
        #print (img_path)
        try:
            image= tf.keras.preprocessing.image.load_img(img_path, target_size=(224,224))
            input_arr = tf.keras.preprocessing.image.img_to_array(image)
            input_arr /= 255.
            input_arr = np.array([input_arr])
            predictions = model.predict(input_arr)
            if np.argmax(predictions) == 1:
                print(img_path, predictions)
        except:
            pass
    break











