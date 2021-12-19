from numpy.core.fromnumeric import ptp
from match_faces import match_image
import pathlib
import os
import tensorflow as tf


def test_image():
    
    # UNKNOWN_FACES_DIR = input("Please type in the full path of the file containing unknown faces")
    UNKNOWN_FACES_DIR = r'C:\Users\gtanwar2\Documents\Mask Detection Project\Cropped_detected Faces'
    print('Processing unknown faces...')

    # Now let's loop over a folder of faces we want to label
    for filename in os.listdir(UNKNOWN_FACES_DIR):

        # Load image
        print(f'Filename {filename}', end='')
        print()
        image_path = os.path.join(UNKNOWN_FACES_DIR,filename)

        # image_path = r'C:\Users\gtanwar2\Documents\Mask Detection Project\Unknown Faces\images.jpg'
        # image_path = input("Please enter the path of the image you want to test: ")
        # image_path = pathlib.Path(image_path)
        cropped_images = match_image(image_path=image_path)
        #Get the folder path and saved model file name from user
        model_path = r'C:\Users\gtanwar2\Documents\Mask Detection Project\mask_detector.model'
        # model_path = input("Please type in the full path of your model:    ")
        # model_path = pathlib.Path(model_path)
        # model_filename = input("Please type in the file name of your model:    ")
        # model_path = os.path.join(modeldirectory,model_filename)
        

        # Load the saved model from disk
        model = tf.keras.models.load_model(model_path)
        
        for image,match in cropped_images:
            
            prediction =model.predict_on_batch(tf.expand_dims(image,axis=0) ).flatten()
            prediction = tf.where(prediction < 0.5, 0, 1)
            # tf.reset_default_graph()

            # with tf.Session() as sess:
            #     init = tf.global_variables_initializer()
            #     sess.run(init)
            #     prediction = sess.run(prediction)
            # print(str(prediction.numpy()))
            print(str(match)+" "+str(prediction.numpy()[0])+"\n")



test_image()