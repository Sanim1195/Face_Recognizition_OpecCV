import os
import cv2 as cv
from retinaface import RetinaFace
from deepface import DeepFace
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score




# 1. Load RetinaFace model:
# img = cv.imread("asset/mjTest1.jpg")
# resp = RetinaFace.detect_faces("asset/mjTest1.jpg")
# get full json output:
""" for face_id, face_data in resp.items():
    print(f"\nFace {face_id}:")
    print(f"Score: {face_data['score'] * 100:.2f}%")
    print(
        f"Facial Area: ({face_data['facial_area'][0]}, {face_data['facial_area'][1]}) - ({face_data['facial_area'][2]}, {face_data['facial_area'][3]})")
    print("Landmarks:")
    for landmark, coordinates in face_data['landmarks'].items():
        print(
            f"  {landmark.capitalize()}: ({coordinates[0]:.2f}, {coordinates[1]:.2f})") """

artist =  ['50cent']   # type: ignore #MJ the GOAT!! , 'Kanye', 'Eminem', 'MichaelJackson'
ROOT_DIR = 'asset/Face_Recon_Dataset'     #Path to image dataset
faces_roi =[]
labels = []
embeddings = []
# now draw a rectangle on the face coordinates
# The facial range has :
# x1, y1) = (28, 51)  # top-left corner
# (x2, y2) = (61, 98)  # bottom-right corner
""" This defines the rectangular bounding box around the detected face.
- x1 (28): left edge of the face
- y1 (51): top edge of the face˜
- x2 (61): right edge of the face
- y2 (98): bottom edge of the face """



def get_roi():
    for artist_name in artist:
        # getting the index of the artist's name
        label = artist.index(artist_name)
        image_folder = os.path.join(ROOT_DIR,artist_name)    # getting to actual folder taht houss the images
        for artist_images in os.listdir(image_folder):       # listing all the images inside that directory
            image = os.path.join(image_folder,artist_images)
            resp = RetinaFace.detect_faces(image)
            # making sure a face exists
            if isinstance(resp,dict):
                img = cv.imread(image)
                for face_id, face_data in resp.items():
                    # print(face_id)
                    # print("x1: ", face_data['facial_area'][0])
                    # print("y1: ", face_data['facial_area'][1])
                    # print("x2: ", face_data['facial_area'][2])
                    # print("y2: ", face_data['facial_area'][3], "\n")
                    # Reading image
                    
                    # detecting faces
                    x1 = face_data['facial_area'][0]
                    y1 = face_data['facial_area'][1]
                    x2 = face_data['facial_area'][2]
                    y2 = face_data['facial_area'][3]
                        
                    # Drawing a bounding box for the face 
                    # faces_rect = cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    face_roi = img[y1:y2,x1:x2]

                    # labelling the cropped roi faces with its name
                    faces_roi.append(face_roi)

                    labels.append(label)
    print(len(faces_roi))
    print(len(labels))
    print("Images lablled and indexed")
    print("Initializing Embedding process.....")
    get_embeddings()

def get_embeddings():
    """ Using deepface to extract embeddings from each facial roi """
    print("Satarting embedding: 🚀🚀 ")
    for roi in faces_roi:
        face_roi_resized = cv.resize(roi, (160, 160))  # Resize the face ROI to 160x160 pixels
        embedding = DeepFace.represent(face_roi_resized, model_name="Facenet")
        print(embedding)
        embeddings.append(embedding)
    print("Vectors stored in list..")


get_roi()

# Time to test and Train this bad boi using svm classifier
# labelling embedding and index to numpy array
X = np.array(embeddings)  #feature
y = np.array(labels)      #label

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an SVM classifier
svm_model = SVC(kernel='linear')  # Linear kernel is a good default for embeddings
svm_model.fit(X_train, y_train)

# Evaluate the model
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"SVM Model accuracy: {accuracy * 100:.2f}%")

