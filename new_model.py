import cv2 as cv
from retinaface import RetinaFace
from deepface import DeepFace
import os

# 1. Load RetinaFace model:
img = cv.imread("asset/mjTest1.jpg")
resp = RetinaFace.detect_faces("asset/mjTest1.jpg")
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

artist: ['50Cent', 'Kanye', 'Eminem', 'Michael Jackson']   #MJ the GOAT!!
image_folder = 'asset/Face_Recon_Dataset'     #Path to image dataset 
faces_roi =[]
label = []
# now draw a rectangle on the face coordinates
# The facial range has :
# x1, y1) = (28, 51)  # top-left corner
# (x2, y2) = (61, 98)  # bottom-right corner
""" This defines the rectangular bounding box around the detected face.
- x1 (28): left edge of the face
- y1 (51): top edge of the face˜
- x2 (61): right edge of the face
- y2 (98): bottom edge of the face """
for artist_name in artist:
    labels = artist.index(artist_name)
    for face_id, face_data in resp.items():
        image_path = os.path.join(image_folder,artist_name)    # getting to actual folder taht houss the images
        for artist_images in os.listdir(image_path):            # listing all the images inside that directory
            print(face_id)
            print("x1: ", face_data['facial_area'][0])
            print("y1: ", face_data['facial_area'][1])
            print("x2: ", face_data['facial_area'][2])
            print("y2: ", face_data['facial_area'][3], "\n")

            x1 = face_data['facial_area'][0]
            y1 = face_data['facial_area'][1]
            x2 = face_data['facial_area'][2]
            y2 = face_data['facial_area'][3]

            faces_rect = cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # labelling the data 
            faces_roi.append((x1,y1,x2,y2))
            label.append(labels)





cv.imshow("Retinaface", img)
cv.waitKey(0)
cv.destroyAllWindows()
