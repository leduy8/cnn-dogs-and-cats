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
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# * Using GPU for processing
# physical_devices = tf.config.experimental.list_physical_devices("GPU")
# #print("Num GPUs Available: ", len(physical_devices))
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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


def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


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

# * Extract batch of images and their corresponding labels
test_imgs, test_labels = next(test_batches)

# plotImages(test_imgs)
print(test_labels)

# ? Get corresponding labels for test dataset
print(test_batches.classes)

# * Predict test dataset
predictions = model.predict(x=test_batches, steps=len(test_batches), verbose=0)
# ? Round predictions to see what the data look like
np.round(predictions)

# * Create a confusion matrix
cm = confusion_matrix(y_true=test_batches.classes,
                      y_pred=np.argmax(predictions, axis=-1))


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# ? Inspect the class so we pass in the correct order
print(test_batches.class_indices)

cm_plot_labels = ['cat', 'dog']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title="Confusion Matrix")
