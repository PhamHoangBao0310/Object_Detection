import config
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def get_data_set_from_directory(data_dir):
    BATCH_SIZE = 128
    IMG_SIZE = (224, 224)
    BUFFER_SIZE = BATCH_SIZE*5

    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
                                              data_dir,
                                              validation_split=0.2,
                                              subset="training",
                                              seed=123,
                                              image_size=IMG_SIZE,
                                              batch_size=BATCH_SIZE)
    validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
                                              data_dir,
                                              validation_split=0.2,
                                              subset="validation",
                                              seed=123,
                                              image_size=IMG_SIZE,
                                              batch_size=BATCH_SIZE)
    return train_dataset, validation_dataset


def model(base="MobileNet"):

  # Data augmentation layer
  data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset= -1)
  ])

  # Base model for transfer learning
  base_model = None
  if base == "MobileNet":
    base_model = tf.keras.applications.MobileNetV2(input_shape=(224,224,3)
                                                  , include_top=False
                                                  , weights="imagenet" )
  else:
    base_model = tf.keras.applications.InceptionV3(input_shape=(224,224,3)
                                                  , include_top=False
                                                  , weights="imagenet" )


  # Flattening
  global_average = tf.keras.layers.GlobalAveragePooling2D()

  # final layer
  dropout_layer = tf.keras.layers.Dropout(0.2)
  prediction_layer = tf.keras.layers.Dense(2)

  inputs = tf.keras.Input(shape=(224, 224, 3))
  x = data_augmentation(inputs)
  x = base_model(x)
  x = global_average(x)
  x = dropout_layer(x)
  outputs = prediction_layer(x)
  model = tf.keras.Model(inputs, outputs)

  return model

class InferenceModel(object):

  model = None

  def __init__(self, model_dir, base):
    # Data augmentation layer
    data_augmentation = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
      tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
      tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset= -1)
    ])

    # Base model for transfer learning
    base_model = None
    if base == "MobileNet":
      base_model = tf.keras.applications.MobileNetV2(input_shape=(224,224,3)
                                                    , include_top=False
                                                    , weights="imagenet" )
    else:
      base_model = tf.keras.applications.InceptionV3(input_shape=(224,224,3)
                                                    , include_top=False
                                                    , weights="imagenet" )
    # Flattening
    global_average = tf.keras.layers.GlobalAveragePooling2D()
    # final layer
    dropout_layer = tf.keras.layers.Dropout(0.2)
    prediction_layer = tf.keras.layers.Dense(2, activation="softmax")

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    x = base_model(x)
    x = global_average(x)
    x = dropout_layer(x)
    outputs = prediction_layer(x)
    self.model = tf.keras.Model(inputs, outputs)

    self.model.load_weights(model_dir)

  def predict(self, image):
    return self.model.predict(image)


if __name__ == '__main__':
  # train_dataset, validation_dataset = get_data_set_from_directory("rcnn\dataset")
  # mymodel =  model()
  # print(mymodel.summary())

  my_parser = argparse.ArgumentParser(description='')
  my_parser.add_argument("-d", "--data_dir", type=str, default="",
                        help="Folder contains your training dataset")
  my_parser.add_argument("-m", "--model", type=str, default="MobileNet",
                        help="The backbone of model. MobileNet of InceptionNet")
  my_parser.add_argument("-ckpt", "--checkpoint", type=str, default="",
                        help="Folder to save your model in .h5 type")

  args = vars(my_parser.parse_args())
  print(args)
