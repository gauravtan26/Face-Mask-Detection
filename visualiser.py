import matplotlib.pyplot as plt
import seaborn as sns
from constants import *
import numpy as np

def visualise_data(dataset):
    class_names = dataset.class_names
    print(class_names)
    plt.figure(figsize=(10,10))
    for images,lables in dataset.take(1):
        for i in range(9):
            ax = plt.subplot(3,3,i+1)
            plt.imshow(images[i].numpy().astype('uint8'))
            plt.title(class_names[int(lables[i])])
            plt.axis("off")
    plt.savefig('Data Sample.png')
    
def draw_count_plot(data,filename):
    plt.figure()
    sns.countplot(data)
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.savefig(filename)

def plot_learning_curves(epochs_range,acc,val_acc,loss,val_loss):
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
    plt.savefig('learning curve.png')


def plot_learning_curve(H):
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("learning_curve.png")

def plot_predicted_data(image,predictions,class_names):
    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(image[i].astype("uint8"))
        plt.title(class_names[int(predictions[i])])
        plt.axis("off")
        plt.savefig("test_data_predictions.png")

