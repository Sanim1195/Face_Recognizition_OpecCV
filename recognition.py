#           This file tests the face_recognition.py model by using it's yml file
import numpy as np
import cv2 as cv

HAAR_CASCADE = cv.CascadeClassifier('har_facedetection.xml')
CELEBRITY_NAMES = ['50cent', 'Eminem', 'IceCube', 'Kanye', 'MichaelJackson']

features = np.load('features.npy', allow_pickle=True)
labels = np.load('labels.npy')

# instatiating the face recpgniser class
face_recognizer = cv.face.LBPHFaceRecognizer_create()
# Reading trained the yml file
face_recognizer.read('faces_trained.yml')

# Reading a test image /Users/sanim/Python/Face_Recognizition_OpecCV/
img = cv.imread("asset/Face_Recon_Dataset/Eminem/Eminem5.jpg")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Person", gray)


# Detect the face in the image
faces_rect = HAAR_CASCADE.detectMultiScale(
    gray, scaleFactor=1.1, minNeighbors=3)

for x, y, w, h in faces_rect:
    faces_roi = gray[y:y+h, x:x+w]

    label, confidence = face_recognizer.predict(faces_roi)
    print(f"Label {CELEBRITY_NAMES[label]} with confidence of  {confidence}")
    cv.putText(img, f"{CELEBRITY_NAMES[label]}", (20, 20),
               cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness=2)
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), thickness=2)
cv.imshow("detected", img)

cv.waitKey(0)
