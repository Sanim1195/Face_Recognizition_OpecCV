# import cv2 as  cv
# import numpy as np
# from retinaface import RetinaFace
# from deepface import DeepFace

# # Step 1: Load the image
# image_path = "asset/Face_Recon_dataset/50cent/50cent3.jpg"
# image_bgr = cv.imread(image_path)

# # Step 2: Detect face using RetinaFace
# faces = RetinaFace.detect_faces(image_bgr)

# if len(faces) > 0:
#     for key, face in faces.items():
#         # Extract the bounding box of the first detected face
#         facial_area = face['facial_area']
#         x, y, w, h = facial_area
        
#         # Crop the face from the original image
#         face_crop_bgr = image_bgr[y:h, x:w]
        
#         # Step 3: Convert the cropped face from BGR to RGB if needed
#         # face_crop_rgb = cv.cvtColor(face_crop_bgr, cv.COLOR_BGR2RGB)
#         # face_crop_rgb = cv.cvtColor(face_crop_bgr)
        
#         # Step 4: Use DeepFace to extract embeddings from the cropped face
#         try:
#             embeddings = DeepFace.represent(face_crop_bgr, model_name='VGG-Face')
#             print("Face embeddings:", embeddings)
#         except ValueError as e:
#             print(f"Error: {e}")
# else:
#     print("No face detected using RetinaFace.")

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

artist =  ['50cent']    #MJ the GOAT!! , 'Kanye', 'Eminem', 'MichaelJackson'
ROOT_DIR = 'asset/Face_Recon_Dataset/50cent/50cent7.png'     #Path to image dataset
faces_roi =[]
labels = []
embeddings = []
def get_roi():
    resp = RetinaFace.detect_faces(ROOT_DIR)
    # making sure a face exists
    if isinstance(resp,dict):
        img = cv.imread(ROOT_DIR)
        for face_id, face_data in resp.items():
            label = artist.index('50cent')
            # detecting faces
            x1 = face_data['facial_area'][0]
            y1 = face_data['facial_area'][1]
            x2 = face_data['facial_area'][2]
            y2 = face_data['facial_area'][3]
                
            # Drawing a bounding box for the face
            faces_rect = cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.imshow("ROI", faces_rect)
            if (cv.waitKey(0) & 0xFF == ord('q')):
                face_roi = img[y1:y2,x1:x2]
                cv.imwrite("ROI.jpg",face_roi)
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
        face_roi_rgb = cv.cvtColor(face_roi_resized, cv.COLOR_BGR2RGB)
        print("The roi of rgb is: /n",face_roi_rgb.shape)
        embedding = DeepFace.represent(face_roi_rgb, model_name="VGG-Face", enforce_detection=False)
        print(embedding)
        embeddings.append(embedding)
    print("Vectors stored in list")


get_roi()
exit()
# Time to test and Train this bad boi using svm classifier
# labelling embedding and index to numpy array

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



