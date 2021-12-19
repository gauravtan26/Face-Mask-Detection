from threading import Thread
import numpy as np
import pathlib
import cv2
import os
import face_recognition
from numpy.lib.type_check import imag
from face_recognition_constants import *
import tensorflow as tf


def get_stored_encodings():
    face_encoding_path = r'C:\Users\gtanwar2\Documents\Mask Detection Project\Face Recognition\known_faces.npy'
    # face_encoding_path = input("Please type in the full path of the file containing known face encodings")
    face_encodings = np.load(pathlib.Path(face_encoding_path), allow_pickle=True)

    known_faces_path = r'C:\Users\gtanwar2\Documents\Mask Detection Project\Face Recognition\known_names.npy'
    # known_faces_path = input("Please type in the full path of the file containing known faces names")
    known_faces_names = np.load(pathlib.Path(known_faces_path), allow_pickle=True)

    return face_encodings, known_faces_names


def match_faces_from_directory():
    
    face_encodings, known_faces_names = get_stored_encodings()
    

    UNKNOWN_FACES_DIR = input("Please type in the full path of the file containing unknown faces")
    UNKNOWN_FACES_DIR = pathlib.Path(UNKNOWN_FACES_DIR)
    
    print('Processing unknown faces...')

    # Now let's loop over a folder of faces we want to label
    for filename in os.listdir(UNKNOWN_FACES_DIR):

        # Load image
        print(f'Filename {filename}', end='')
        image = face_recognition.load_image_file(f'{UNKNOWN_FACES_DIR}/{filename}')

        # This time we first grab face locations - we'll need them to draw boxes
        locations = face_recognition.face_locations(image, model=MODEL)

        # Now since we know loctions, we can pass them to face_encodings as second argument
        # Without that it will search for faces once again slowing down whole process
        encodings = face_recognition.face_encodings(image, locations)

        # We passed our image through face_locations and face_encodings, so we can modify it
        # First we need to convert it from RGB to BGR as we are going to work with cv2
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        print(f', found {len(encodings)} face(s)')
        for face_encoding, face_location in zip(encodings, locations):

            # We use compare_faces (but might use face_distance as well)
            # Returns array of True/False values in order of passed known_faces
            results = face_recognition.compare_faces(face_encodings, face_encoding, TOLERANCE)
            # print(results)
            match = None
            if True in results:
                match = known_faces_names[results.index(True)]
                top_left = (max(0,face_location[3]-50),max(0,face_location[0]-50))
                bottom_right = (face_location[1]+50,face_location[2]+50)

                # print("top_left",top_left)
                # print("bottom_right",bottom_right)
                # print(image)
                cropped_image = image[max(0,face_location[0]-50):face_location[2]+50,max(0,face_location[3]-50):face_location[1]+50]
                print("Cropped Image")
                # print(cropped_image)
                cv2.imwrite(os.path.join(r'C:\Users\gtanwar2\Documents\Mask Detection Project\Cropped_detected Faces',filename+"_cropped.jpg"),cropped_image)

                color = [0,255,0]
                # Paint frame
                cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

                # Now we need smaller, filled grame below for a name
                # This time we use bottom in both corners - to start from bottom and move 50 pixels down
                top_left = (face_location[3], face_location[2]+50)
                bottom_right = (face_location[1]+50, face_location[2] + 22+50)

                # Paint frame
                cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)

                # Wite a name
                cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15+50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), FONT_THICKNESS)


       
        # Show image
        cv2.imshow(filename, image)
        cv2.waitKey(0)
        cv2.destroyWindow(filename)

from time import sleep
def face_recognition_videos():
    video = cv2.VideoCapture(r'C:\Users\gtanwar2\Pictures\Video Projects\face_recotest.mp4') 
    face_encodings, known_faces_names = get_stored_encodings()
    # video = cv2.VideoCapture(0)
    print('Processing unknown faces...')
    while True:
        ret,image = video.read()
        locations = face_recognition.face_locations(image, model=MODEL)
        encodings = face_recognition.face_encodings(image, locations)
        print(f', found {len(encodings)} face(s)')
        for face_encoding, face_location in zip(encodings, locations):
            results = face_recognition.compare_faces(face_encodings, face_encoding, TOLERANCE)
            match = None
            if True in results:
                match = known_faces_names[results.index(True)]
                print(match)
                # top_left = (max(0,face_location[3]-50),max(0,face_location[0]-50))
                # bottom_right = (face_location[1]+50,face_location[2]+50)                
                # color = [0,255,0]
                # cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)
                # top_left = (face_location[3], face_location[2]+50)
                # bottom_right = (face_location[1]+50, face_location[2] + 22+50)
                # cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
                # cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15+50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), FONT_THICKNESS)     
        # Show image
        # cv2.imshow("video", image)
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break
        # # k = cv2.waitKey(30) & 0xff
        # # if k==27:
        # #     break
        # sleep(10000)
        # cv2.destroyWindow("video")
        
# face_recognition_videos()


def watch_video():
    # video = cv2.VideoCapture(r'C:\Users\gtanwar2\Documents\Mask Detection Project\Video Data\SACHIN TENDULKAR - Iconic Speech.mp4') 
    video = cv2.VideoCapture(0)
    while True:
        ret,image = video.read()
        cv2.imshow("video", image)
        # Stop if escape key is pressed
        k = cv2.waitKey(30) & 0xff
        if k==27:
            break
# watch_video()


def match_image(image_path):
    face_encodings, known_faces_names = get_stored_encodings()

    image = face_recognition.load_image_file(image_path)
    locations = face_recognition.face_locations(image, model=MODEL)
    encodings = face_recognition.face_encodings(image, locations)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    cropped_images = []
    for face_encoding, face_location in zip(encodings, locations):

        # We use compare_faces (but might use face_distance as well)
        # Returns array of True/False values in order of passed known_faces
        results = face_recognition.compare_faces(face_encodings, face_encoding, TOLERANCE)
        # print(results)
        match = None
        if True in results:
            match = known_faces_names[results.index(True)]
            top_left = (max(0,face_location[3]-50),max(0,face_location[0]-50))
            bottom_right = (face_location[1]+50,face_location[2]+50)

            # print("top_left",top_left)
            # print("bottom_right",bottom_right)
            # print(image)
            cropped_image = image[max(0,face_location[0]-50):face_location[2]+50,max(0,face_location[3]-50):face_location[1]+50]
            # # print(cropped_image)
            # cv2.imwrite(os.path.join(r'C:\Users\gtanwar2\Documents\Mask Detection Project\Cropped_detected Faces',filename+"_cropped.jpg"),cropped_image)

            # color = [0,255,0]
            # # Paint frame
            # cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

            # # Now we need smaller, filled grame below for a name
            # # This time we use bottom in both corners - to start from bottom and move 50 pixels down
            # top_left = (face_location[3], face_location[2]+50)
            # bottom_right = (face_location[1]+50, face_location[2] + 22+50)

            # # Paint frame
            # cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)

            # # Wite a name
            # cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15+50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), FONT_THICKNESS)
            cropped_images.append((tf.image.resize(cropped_image, (224, 224)) ,match))

    return cropped_images

face_recognition_videos()