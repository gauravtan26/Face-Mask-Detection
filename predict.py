import sys
import os
import glob
import tensorflow as tf
import dataset_generator
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,plot_confusion_matrix
from visualiser import plot_predicted_data

def predict_on_test_data():
    
    #Get the folder path and saved model file name from user
    modeldirectory = input("Please type in the full path of the folder containing your model:    ")
    model_filename = input("Please type in the file name of your model:    ")
    model_path = os.path.join(modeldirectory,model_filename)
    

    # Load the saved model from disk
    model = tf.keras.models.load_model(model_path)
    model.summary()

    test_dataset = dataset_generator.generate_test_data()

    # Retrieve a batch of images from the test set
    image_batch, label_batch = test_dataset.as_numpy_iterator().next()
    predictions_batch = model.predict_on_batch(image_batch).flatten()

    # Apply a sigmoid since our model returns logits
    # predictions = tf.nn.sigmoid(predictions)
    predictions_batch = tf.where(predictions_batch < 0.5, 0, 1)
    plot_predicted_data(image_batch,predictions_batch,test_dataset.class_names)
    

    ground_truth = []
    predictions = []

    for images, labels_batch in test_dataset:
        ground_truth.extend(list(labels_batch.numpy().squeeze()))
        predictions_batch = model.predict_on_batch(images).flatten()
        predictions_batch = tf.where(predictions_batch < 0.5, 0, 1)
        predictions.extend(predictions_batch.numpy().squeeze())

    print(classification_report(ground_truth,predictions))



predict_on_test_data()