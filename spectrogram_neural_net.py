import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys

import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib


def build_dataframe(csv_file):
    # make dataframe from csv
    print('Making dataframe from file:', csv_file)
    d_frame = pd.read_csv(csv_file, skiprows=1, header=None)
    print('Dataframe complete.')
    print('Number of rows: ', d_frame.shape[0])
    print('Number of columns: ', d_frame.shape[1])
    return d_frame

def write_csv(id_list, class_list, filename):
    print('Writing csv')
    filename = filename + '.csv'
    indices = np.asarray(id_list[0])

    # concatenate arrays, transpose, and save as csv
    full_array = np.concatenate(([indices], [class_list]), axis=0)
    full_array_transpose = np.transpose(full_array)
    full_dataframe = pd.DataFrame(full_array_transpose, columns = ['id','genre'])
    full_dataframe.to_csv(filename, index=False)

# define directories for spectrograms
data_dir = pathlib.Path('./spectrograms/train/')
test_data_dir = pathlib.Path('./spectrograms/test/')

# set batch size and image dimensions
batch_size = 32
img_height = 261
img_width = 525

# the function image_dataset_from_directory() gets the class label
# from the subdirectories within the training directory

# define training dataset from training images
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=8,
    image_size=(img_height, img_width),
    batch_size=batch_size
    )

# define validation dataset from training images
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=8,
    image_size=(img_height, img_width),
    batch_size=batch_size
    )

# define training dataset as all training images
all_train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

# define test dataset
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_data_dir,
    labels=None,
    batch_size=batch_size,
    image_size=(img_height, img_width),
    shuffle=False
)

# print to show class names
class_names = train_ds.class_names
print('Classes:', class_names)

# image buffer to increase performance
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
all_train_ds = all_train_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# define classes
num_classes = len(class_names)

# convolutional model: convolutional layers followed by max pooling, relu activation functions
# flattening layer, dense layers with relu activation functions
model = Sequential([
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    layers.Conv2D(16, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv1D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv1D(64, 3,  activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.3),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(num_classes)
])

# adam optimizer
# sparse categorical cross entropy loss for multiple classes
# metrics parameter to display accuracy during epoch
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# show model summary
model.summary()

if len(sys.argv) == 1:
    # train model with 80/20 split for validation purposes
    epochs=60
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)


    # plot accuracy and loss
    accuracy = history.history['accuracy']
    loss = history.history['loss']
    val_accuracy = history.history['val_accuracy']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, accuracy, label='Training Accuracy')
    plt.plot(epochs_range, val_accuracy, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    print('Making predictions on test data...')
    result_array = model.predict(test_ds)

if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] == '-p'):
    # executes with -p to indicate prediction only

    # build df of test data filenames
    test_df = build_dataframe('data/test_idx.csv')

    # fit all training data
    model.fit(all_train_ds, epochs=60)

    # predict on test data
    print('Making predictions on test data...')
    result_array = model.predict(test_ds)

    # use filenames and results to make csv of predictions
    write_csv(test_df.iloc[:, 0:], np.argmax(result_array, axis=1), 'spect_nn_submission')