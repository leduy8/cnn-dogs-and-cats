from plot_confusion_matrix import plot_confusion_matrix
from plotImages import plotImages
from get_batches import get_train_batches, get_valid_batches, get_test_batches

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from sklearn.metrics import confusion_matrix
import itertools
import os
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

train_batches = get_train_batches()
valid_batches = get_valid_batches()
test_batches = get_test_batches()

# * Visualizing data
# imgs, labels = next(train_batches)


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


# ? Inspect the class so we pass in the correct order
print(test_batches.class_indices)

cm_plot_labels = ['cat', 'dog']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title="Confusion Matrix")
