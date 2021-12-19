import tensorflow as tf
from tensorflow.python.keras.engine import training
from constants import *
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from visualiser import *
from tensorflow.keras import layers
from sklearn.preprocessing import LabelBinarizer


def generate_data(data_dir):

# Generate training and validation tf.data.Dataset from image files in the directory.

  data_dir = pathlib.Path(data_dir)
  train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  label_mode = 'binary',
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

  validation_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='validation',
    label_mode = 'binary',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)
  
  visualise_data(train_ds)
  
  train_label = []
  validation_label = [] 
  for _, labels_batch in train_ds:
    train_label.extend(list(labels_batch.numpy().squeeze()))

  for _, labels_batch in validation_ds:
    validation_label.extend(list(labels_batch.numpy().squeeze()))
  

  # draw count plots of training and validation data
  draw_count_plot(train_label,"training_count_plot.png")
  draw_count_plot(validation_label,"validation_count_plot.png")
  
  return train_ds,validation_ds


def generate_test_data():
  
  #Get the data directory path 
  data_directory = input("Please type in the full path of the folder containing your test data:    ")
  data_dir = pathlib.Path(data_directory)
  test_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  label_mode = 'binary',
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

  return test_ds