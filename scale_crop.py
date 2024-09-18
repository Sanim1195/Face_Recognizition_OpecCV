import cv2 as cv

image = cv.imread('asset/mjTest1.jpg')
HAAR_CASCADE = cv.CascadeClassifier(
    'har_facedetection.xml')   # Path to haar cascade algo

# scale image:
scale_factor = 0.5  # and 3.0
new_size = (int(image.shape[1] * scale_factor),
            int(image.shape[0] * scale_factor))
resized_image = cv.resize(image, new_size)

# crop image
# finding face in the image
faces = HAAR_CASCADE.detectMultiScale(
    image, scaleFactor=2.1, minNeighbors=9)

# getting the coordinates of ROI from the image
for (x, y, w, h) in faces:
    faces_roi = faces[y:y+h, x:x+w]
    draw = cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
    print(x, y, w, h)
    # now cropping the roi

cv.imshow("Rescaled Image", image)
cv.waitKey(0)
cv.destroyAllWindows()
