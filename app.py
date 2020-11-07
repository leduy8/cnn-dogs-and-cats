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

# * Disable GPU for tensorflow
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# * Using GPU for processing
# physical_devices = tf.config.experimental.list_physical_devices("GPU")
# #print("Num GPUs Available: ", len(physical_devices))
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

# * Pickup images of dogs and cats and put it in the corresponding folders
# os.chdir("data/dogs-vs-cats")
# if os.path.isdir("train/dog") is False:
#     os.makedirs("train/dog")
#     os.makedirs("train/cat")
#     os.makedirs("valid/dog")
#     os.makedirs("valid/cat")
#     os.makedirs("test/dog")
#     os.makedirs("test/cat")

#     for i in random.sample(glob.glob("cat*"), 500):
#         shutil.move(i, 'train/cat')
#     for i in random.sample(glob.glob("dog*"), 500):
#         shutil.move(i, 'train/dog')
#     for i in random.sample(glob.glob("cat*"), 100):
#         shutil.move(i, 'valid/cat')
#     for i in random.sample(glob.glob("dog*"), 100):
#         shutil.move(i, 'valid/dog')
#     for i in random.sample(glob.glob("cat*"), 50):
#         shutil.move(i, 'test/cat')
#     for i in random.sample(glob.glob("dog*"), 50):
#         shutil.move(i, 'test/dog')
# os.chdir("../../")

train_path = 'data/dogs-vs-cats/train'
valid_path = 'data/dogs-vs-cats/valid'
test_path = 'data/dogs-vs-cats/test'

# * Create batches of normalized tensor image data for train
# ? Use preprocessing model vgg16
# ? Target size for image resolution
train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(
    directory=train_path, target_size=(224, 224), classes=['cat', 'dog'], batch_size=10)

valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(
    directory=valid_path, target_size=(224, 224), classes=['cat', 'dog'], batch_size=10)

test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(
    directory=test_path, target_size=(224, 224), classes=['cat', 'dog'], batch_size=10, shuffle=False)

# * Visualizing data
# imgs, labels = next(train_batches)


# def plotImages(images_arr):
#     fig, axes = plt.subplots(1, 10, figsize=(20, 20))
#     axes = axes.flatten()
#     for img, ax in zip(images_arr, axes):
#         ax.imshow(img)
#         ax.axis('off')
#     plt.tight_layout()
#     plt.show()


# print(labels)
# plotImages(imgs)

# * Build a CNN model
model = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu',
           padding='same', input_shape=(224, 224, 3)),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Flatten(),
    Dense(units=2, activation='softmax')
])

# model.summary()

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy', metrics=['accuracy'])

# * Train CNN model
# ? steps_per_epoch = num of dataset / batch_size should be equal to 100. But we specify batch size of train_batches is 10 so it's equal to 100, same as the steps_per_epoch so we only need to use len() for more general (note that this is a coincidence, if not, then we set as specific number)
model.fit(x=train_batches, steps_per_epoch=len(train_batches),
          validation_data=valid_batches, validation_steps=len(valid_batches), epochs=10, verbose=2)
