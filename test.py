
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


def center_crop(x, center_crop_size, **kwargs):
    centerw, centerh = x.shape[0]//2, x.shape[1]//2
    halfw, halfh = center_crop_size[0]//2, center_crop_size[1]//2
    return x[centerw-halfw:centerw+halfw,centerh-halfh:centerh+halfh, :]

def predict_10_crop(img, top_n=5, plot=False, preprocess=True):
    flipped_X = np.fliplr(img)
    crops = [
        img[:224,:224, :], # Upper Left
        img[:224, img.shape[1]-224:, :], # Upper Right
        img[img.shape[0]-224:, :224, :], # Lower Left
        img[img.shape[0]-224:, img.shape[1]-224:, :], # Lower Right
        center_crop(img, (224, 224)),

        flipped_X[:224,:224, :],
        flipped_X[:224, flipped_X.shape[1]-224:, :],
        flipped_X[flipped_X.shape[0]-224:, :224, :],
        flipped_X[flipped_X.shape[0]-224:, flipped_X.shape[1]-224:, :],
        center_crop(flipped_X, (224, 224))
    ]
    if preprocess:
        crops = [preprocess_input(x.astype('float32')) for x in crops]

    if plot:
        fig, ax = plt.subplots(2, 5, figsize=(10, 4))
        ax[0][0].imshow(crops[0])
        ax[0][1].imshow(crops[1])
        ax[0][2].imshow(crops[2])
        ax[0][3].imshow(crops[3])
        ax[0][4].imshow(crops[4])
        ax[1][0].imshow(crops[5])
        ax[1][1].imshow(crops[6])
        ax[1][2].imshow(crops[7])
        ax[1][3].imshow(crops[8])
        ax[1][4].imshow(crops[9])

    y_pred = model.predict(np.array(crops))
    y_pred = np.mean(y_pred, axis=0)
    preds = np.argmax(y_pred, axis=-1)
    top_n_preds= np.argpartition(y_pred, -top_n)[-top_n:]
    top_n_probs = [y_pred[ix] for ix in top_n_preds]
    sorted_index = np.argsort(top_n_probs)[::-1]
    return preds, np.array(top_n_preds)[sorted_index], np.array(top_n_probs)[sorted_index]


conv_base = EfficientNetB0(include_top = False,
                      weights = "imagenet",
                      input_tensor = None,
                      input_shape = (224, 224, 3),
                      classes = 101)

model = Sequential()
model.add(conv_base)

model.add(tf.keras.layers.GlobalMaxPooling2D(name='max_pool'))
model.add(Dropout(0.2, name='top_dropout'))
model.add(Dense(101,
                 activation='softmax',
                 name='fc_out'))

model.load_weights("food101.h5")

root = "/home/eduardo-ucn/Documents/datasets/food-101/meta/"

test_df  = pd.read_csv(root+"testdf.csv",dtype=str)
valid_datagen = ImageDataGenerator(rescale=1./255)

test_generator=valid_datagen.flow_from_dataframe(dataframe=test_df,
                                                directory=None,
                                                x_col="id",
                                                y_col="label",
                                                batch_size=32,
                                                seed=42,
                                                shuffle=False,
                                                class_mode="categorical",
                                                validate_filenames=False)

STEP_SIZE_TEST  = test_generator.n//test_generator.batch_size
STEP_SIZE_TEST = STEP_SIZE_TEST + 1 if test_generator.n % test_generator.batch_size != 0 else STEP_SIZE_TEST

count = 0
top1_acc = 0
topn_acc = 0
total = 0
for batch in test_generator:
    #print(np.shape(batch[0]), np.shape(batch[1]))
    count += 1

    if STEP_SIZE_TEST == count:
        data = zip(batch[0][:test_generator.n-(STEP_SIZE_TEST-1)*test_generator.batch_size],batch[1][:test_generator.n-(STEP_SIZE_TEST-1)*test_generator.batch_size])
    else:
        data = zip(batch[0],batch[1])

    for img, label in data:
        pred_label = predict_10_crop(img, top_n=5, plot=False, preprocess=False)
        gt_label = np.argmax(label)
        if gt_label == pred_label[0]:
            top1_acc +=1
        if gt_label in pred_label[1]:
            topn_acc += 1
        total += 1
    print(total)
    if STEP_SIZE_TEST == count:
        break

print(top1_acc/float(total))
print(topn_acc/float(total))

