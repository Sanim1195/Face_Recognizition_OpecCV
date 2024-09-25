import cv2 as cv
from retinaface import RetinaFace

# Initializing RetinaFace model choose 1: [onnx, pytorch, tenserflow,keras]
# model = RetinaFace(backend='onnx')
# detector = RetinaFace()
HAAR_CASCADE = cv.CascadeClassifier('har_facedetection.xml')   # Path to haar cascade algo

# Specifying which camera to use
video = cv.VideoCapture(0)

# checking if video object was sucessfully captured
if not video.isOpened():
    print("Error can't open video cam")
    exit()

# if the video cam opens
while True:
    # capture video frame by frame
    ret, frame = video.read()
    # print(frame)
    # check if teh frame was not captured
    if not ret:
        print("Error cannot read frame")
        break
    # If everything is fine detect face from model
    # faces = RetinaFace.detect_faces(frame)
    # This is still shot hahahah:) 
    faces_rect = HAAR_CASCADE.detectMultiScale(
                frame, scaleFactor=1.5, minNeighbors=5)
    # draw a rectangle over it
    # Draw rectangles around detected faces
    for (x, y, w, h) in faces_rect:
        faces_roi = frame[y:y+h, x:x+w]
        cv.rectangle(frame, (x,x+w),(y,y+h), (0, 255, 0), 3)

    # display the frame in window
    cv.imshow("Video", frame)
    # Break the loop if 'q' is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the window
video.release()
cv.destroyAllWindows()
