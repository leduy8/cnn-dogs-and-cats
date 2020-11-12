from pathlib import Path
from plotImages import plotImages
from plot_confusion_matrix import plot_confusion_matrix
from get_batches import get_train_batches, get_valid_batches, get_test_batches

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

Path("./models").mkdir(parents=True, exist_ok=True)

# * Disable GPU for tensorflow
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# * Using GPU for processing
# physical_devices = tf.config.experimental.list_physical_devices("GPU")
# #print("Num GPUs Available: ", len(physical_devices))
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

train_batches = get_train_batches()
valid_batches = get_valid_batches()
test_batches = get_test_batches()

# * Get the original trained model with its saved weights and other params
vgg16_model = tf.keras.applications.vgg16.VGG16()
# ? See type of vgg16_model: Functional from Functional API from Keras
# print(type(vgg16_model))
# vgg16_model.summary()

# * Create a Sequential for simplicity and add in the all the hidden layers from vgg16 model
model = Sequential()
for layer in vgg16_model.layers[:-1]:
    model.add(layer)

# * Freeze all the hidden model for when we train, it's not update the weights
for layer in model.layers:
    layer.trainable = False

# * Add the last layer with 2 nodes to identify cats and dogs
model.add(Dense(units=2, activation='softmax'))

# model.summary()

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x=train_batches, steps_per_epoch=len(train_batches),
          validation_data=valid_batches, validation_steps=len(valid_batches), epochs=5, verbose=2)

# * Save model(the architecture, the weights, the optimizer, the state of the optimizer, the learning rate, the loss, etc.) to a .h5 file
# ? If found a model, delete it and save a new one
if os.path.isfile("models/fine-tune_dogs_vs_cats.h5") is True:
    os.remove("models/fine-tune_dogs_vs_cats.h5")
model.save(r"models/fine-tune_dogs_vs_cats.h5")

# predictions = model.predict(x=test_batches, steps=len(test_batches), verbose=0)

# cm = confusion_matrix(y_true=test_batches.classes,
#                       y_pred=np.argmax(predictions, axis=-1))
# cm_plot_lables = ['cat', 'dog']
# plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion matrix')
