# The face recognition model

import os
import cv2 as cv
import numpy as np

CELEBRITY_NAMES = ['50cent', 'Eminem', 'IceCube', 'Kanye', 'MichaelJackson']
HAAR_CASCADE = cv.CascadeClassifier('har_facedetection.xml')   # Path to haar cascade algo
DIR = "./asset/Face_Recon_Dataset"   # Path to image dataset

# For training set
features = []  # image arrays of faces
labels = []  # correspoding int labels to the CELEBRITY_NAMES for better indexing


def create_train():
    """ Traverses through the folder, adds int labels to folders, finds image, reads image
        detects face, draws rectangle on to of the detected face.
         Our folder structure:
        asset-> Face_Recon_Dataset->[50cebt,Eminem,IceCube,Kanye,MichaelJackson]->[actual images]  """
    for person in CELEBRITY_NAMES:
        path = os.path.join(DIR, person)
        print("Folder path : ", path)
        label = CELEBRITY_NAMES.index(person)  # Generating int labels
        print("It's labelled as : ", label)
        # traversing through the folder and getting actual file path
        for image in os.listdir(path):
            image_path = os.path.join(path, image)
            print("Image path : ", image_path)

            # Reading the image
            img_array = cv.imread(image_path)
            gray_image = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            # finding face in the image
            faces_rect = HAAR_CASCADE.detectMultiScale(
                gray_image, scaleFactor=1.1, minNeighbors=3)

            # Drawing a rectangle on the above coordinates
            for (x, y, w, h) in faces_rect:
                faces_roi = gray_image[y:y+h, x:x+w]
                # appending the rectangle coordinates to the features list
                features.append(faces_roi)
                labels.append(label)  # appending int labels to the list

        print("\n")


create_train()
print("trainning done -------------------------------------------------------->")
# print("the length of the fetures list is: ", len(features))
# print("the length of the labels list is: ", len(labels))

# Converting the lists into a numpy array
features = np.array(features, dtype='object')
labels = np.array(labels)

# Using the features and labels list to train our recogniser
""" See each features contains the region of interest(roi) for an image (the width and height of the face)
And the labels list labels these roi by integers(0,1,2,3,4) whcich we know is the index for our
celebrity_names list """

# Instatiate the face recognizer from face.LBPHFaceRecognizer_create() class
face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Train your recogniser on your features and labels list
face_recognizer.train(features,labels)

# Saving the trained data as a yml file so we can access the trained from anywhere.
face_recognizer.save("faces_trained.yml")

# Saving the features and labels list
np.save("features.npy", features)
np.save("labels.npy", labels)
