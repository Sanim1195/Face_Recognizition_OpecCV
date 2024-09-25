import cv2
from deepface import DeepFace

# Load the pre-trained face detection model from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Read the input image
input_image_path = 'asset/mjTest1.jpg'
image = cv2.imread(input_image_path)

# Convert the image to grayscale (required for face detection)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Iterate through detected faces
for (x, y, w, h) in faces:
    # Crop the face region
    face_roi = image[y:y+h, x:x+w]

    # Analyze the face using DeepFace
    result = DeepFace.analyze(face_roi, actions=['emotion', 'age', 'gender'])

    # Print the analysis results
    print(result)
    # print(f"Emotion: {result['emotion']}, Age: {result['age']}, Gender: {result['gender']}")

# Display the image with rectangles around detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Show the final image
cv2.imshow('Detected Faces', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
