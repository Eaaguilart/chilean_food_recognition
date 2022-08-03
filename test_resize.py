
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

from tensorflow.keras.applications import EfficientNetB0, DenseNet169, DenseNet121, VGG16, ResNet152, ResNet152V2, InceptionV3, ResNet50,ResNet50V2


conv_base = EfficientNetB0(include_top = False,
                      weights = "imagenet",
                      input_tensor = None,
                      input_shape = (224, 224, 3),
                      classes = 256)

model = Sequential()
model.add(conv_base)

model.add(tf.keras.layers.GlobalMaxPooling2D(name='max_pool'))
model.add(Dropout(0.2, name='top_dropout'))
model.add(Dense(256,
                 activation='softmax',
                 name='fc_out'))

model.load_weights("uecfood256.h5")
#model.load_weights("chileanfood64_b3.h5")

#root = "/home/eduardo-ucn/Documents/datasets/ChileanFood64/"

#test_df  = pd.read_csv(root+"testLabels.csv",dtype=str)


root = "/home/eduardo-ucn/Documents/datasets/uecfood256/meta/"
'''
with open(root+"valdf_0.csv", "r") as f:
    nlines = [f.readline()]
    for line in f:
        if "hot_dog" in line:
            nlines.append(line)
with open(root+"testdf_hd.csv", "w") as f:
    f.writelines(nlines)
'''
test_df  = pd.read_csv(root+"valdf_0.csv",dtype=str)


valid_datagen = ImageDataGenerator(rescale=1./255)

test_generator=valid_datagen.flow_from_dataframe(dataframe=test_df,
                                                directory=root,
                                                x_col="id",
                                                y_col="label",
                                                batch_size=32,
                                                seed=42,
                                                shuffle=False,
                                                class_mode="categorical",
                                                validate_filenames=False,
                                                target_size=(224,224))

print(test_generator.class_indices)

'''
STEP_SIZE_TEST  = test_generator.n//test_generator.batch_size
STEP_SIZE_TEST = STEP_SIZE_TEST + 1 if test_generator.n % test_generator.batch_size != 0 else STEP_SIZE_TEST

count = 0
top1_acc = 0
topn_acc = 0
total = 0
top_n = 5
for batch in test_generator:
    count += 1
    total_data = test_generator.batch_size if STEP_SIZE_TEST != count else test_generator.n-(STEP_SIZE_TEST-1)*test_generator.batch_size
    data = zip(batch[0][:total_data],batch[1][:total_data])
    resize_batch = batch[0][:total_data]
    gt_label = np.argmax(batch[1][:total_data],-1)
    y_pred = model.predict(resize_batch)
    preds = np.argmax(y_pred, axis=-1)
    print(gt_label, preds)
    top1_acc += sum(preds == gt_label)
    total += total_data
    top_n_preds= np.argpartition(y_pred, -top_n)[:,-top_n:]
    for gt, topn_pred in zip(gt_label, top_n_preds):
        if gt in topn_pred:
            topn_acc +=1
    print(total)
    if STEP_SIZE_TEST == count:
        break



print(top1_acc/float(total))
print(topn_acc/float(total))
'''
