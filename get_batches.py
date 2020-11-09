import os
import shutil
import random
import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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


def get_train_batches():
    train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(
        directory=train_path, target_size=(224, 224), classes=['cat', 'dog'], batch_size=10)
    return train_batches


def get_valid_batches():
    valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(
        directory=valid_path, target_size=(224, 224), classes=['cat', 'dog'], batch_size=10)
    return valid_batches


def get_test_batches():
    test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(
        directory=test_path, target_size=(224, 224), classes=['cat', 'dog'], batch_size=10, shuffle=False)
    return test_batches
