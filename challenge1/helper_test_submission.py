import os
import time
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator

tfk = tf.keras
tfkl = tf.keras.layers
seed = 20

dataset_path = '../Challenge/training'
test_path = '../Challenge/test'

img_h, img_w = (256, 256)
isTest = True

labels = ['Apple','Blueberry','Cherry','Corn','Grape','Orange','Peach','Pepper','Potato','Raspberry','Soybean','Squash','Strawberry','Tomato']
validation_split = 0.08

def load_data(dontUseFun=True, fun=None):
  image_generator = ImageDataGenerator(preprocessing_function = fun, validation_split=validation_split)  
  if dontUseFun:
    image_generator = ImageDataGenerator(validation_split=validation_split)  


  # Obtain a data generator with the 'ImageDataGenerator.flow_from_directory' method
  train_gen = image_generator.flow_from_directory(directory=dataset_path,
                                                target_size=(img_h, img_w),
                                                color_mode='rgb',
                                                classes=labels, # can be set to None
                                                class_mode='categorical',
                                                subset='training',
                                                batch_size=batch_size,
                                                shuffle=True,
                                                seed=seed)

  valid_gen = image_generator.flow_from_directory(directory=dataset_path,
                                                target_size=(img_h, img_w),
                                                color_mode='rgb',
                                                classes=labels, # can be set to None
                                                class_mode='categorical',
                                                subset='validation',
                                                batch_size=batch_size,
                                                shuffle=False,
                                                seed=seed)
  test_gen = None
  if isTest:
    test_image_gen = ImageDataGenerator(preprocessing_function = fun)
    if dontUseFun:
      test_image_gen = ImageDataGenerator()
    test_gen = test_image_gen.flow_from_directory(directory=test_path,
                                                target_size=(img_h, img_w),
                                                color_mode='rgb',
                                                classes=labels, # can be set to None
                                                class_mode='categorical',
                                                batch_size=batch_size,
                                                shuffle=False,
                                                seed=seed)
  return train_gen, valid_gen, test_gen


from model import model
a = model("./")

batch_size=1
train_gen, valid_gen, test_gen = load_data(True)
n = next(test_gen)
num = 0
d =0
while n:
    pred, target = n[0], n[1]
    # print(pred.shape)
    if a.predict(pred) == tf.argmax(target, axis=-1): num+=1
    d+=1
    if d % 64 == 0: print(d, ' iteration:', num/d)
    n = next(test_gen)
