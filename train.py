# import the necessary packages
import dataset_generator
from constants import *
import visualiser
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
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras import layers
from model import create_model



def train(model):
    # model.compile()
    history=model.fit()
    display_results(history,range(epochs))
    return history


def display_results(history,epochs_range):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss'] 
    visualiser.plot_learning_curves(epochs_range,acc,val_acc,loss,val_loss)



def main():
    
    # Read the dataset from disk
    train_ds,validation_ds=dataset_generator.generate_data(data_dir)

    # configure the dataset for better performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(buffer_size=100).prefetch(buffer_size=AUTOTUNE)
    validation_ds = validation_ds.cache().prefetch(buffer_size=AUTOTUNE)
     
    model = create_model()
    # compile our model
    print("[INFO] compiling model...")
    opt = Adam(lr=init_lr, decay=init_lr / epochs)
    model.compile(loss="binary_crossentropy", optimizer=opt,
        metrics=["accuracy"])

    # train the head of the network
    print("[INFO] training head...")
    
    # H = model.fit(
    #     aug.flow(train_ds, batch_size=batch_size),
    #     # steps_per_epoch=len(trainX) // BS,
    #     validation_data=validation_ds,
    #     # validation_steps=len(testX) // BS,
    #     epochs=epochs)

    h =model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=epochs
    )

    # serialize the model to disk
    print("[INFO] saving mask detector model...")
    model.save("mask_detector.model", save_format="h5")

    visualiser.plot_learning_curve(h)


if __name__=="__main__":
    main()