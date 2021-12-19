import face_recognition
import os
import cv2
import numpy as np
    

def get_known_faces_encodings():

    print('Loading known faces...')
    known_faces = []
    known_names = []

    #Get the folder path and saved model file name from user
    KNOWN_FACES_DIR = input("Please type in the full path of the directory having known faces:    ")

    for name in os.listdir(KNOWN_FACES_DIR):

        # Next we load every file of faces of known person
        for filename in os.listdir(f'{KNOWN_FACES_DIR}/{name}'):

            # Load an image
            image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{name}/{filename}')

            # Get 128-dimension face encoding
            # Always returns a list of found faces, for this purpose we take first face only (assuming one face per image as you can't be twice on one image)
            encoding = face_recognition.face_encodings(image)[0]

            # Append encodings and name
            known_faces.append(encoding)
            known_names.append(name)

    np.save('known_faces.npy', known_faces, allow_pickle=True)
    np.save('known_names.npy', known_names, allow_pickle=True)
    
get_known_faces_encodings()