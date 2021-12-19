 # Define a simple sequential model
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
import constants

def create_model():

    # load the MobileNetV2 network, ensuring the head FC layer sets are
    # left off
    baseModel = MobileNetV2(weights="imagenet", include_top=False,
        input_tensor=Input(shape=(constants.img_width, constants.img_height, 3)))

    # loop over all layers in the base model and freeze them so they will
    # *not* be updated during the first training process
    for layer in baseModel.layers:
        layer.trainable = False

    # define the model to be used, using augmentation as the 
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Resizing(224,224))
    model.add(tf.keras.layers.Rescaling(1./127.5, offset=-1))
    model.add(layers.RandomFlip("horizontal"))
    model.add(layers.RandomZoom(0.15))
    model.add(layers.RandomRotation(0.2))
    model.add(baseModel)
    model.add(AveragePooling2D(pool_size=(7, 7)))
    model.add(Flatten(name="flatten"))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))

    return model