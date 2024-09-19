import cv2 as cv
from retinaface import RetinaFace

# Initializing RetinaFace model choose 1: [onnx, pytorch, tenserflow,keras]
# model = RetinaFace(backend='onnx')
detector = RetinaFace()
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
    print(frame)
    # check if teh frame was not captured
    if not ret:
        print("Error cannot read frame")
        break
    # If everything is fine detect face from model
    # faces = model.predict(frame)
    faces = detector.detect_faces(frame)
    # draw a rectangl;e over it
    # Draw rectangles around detected faces
    for face in faces:
        x1, y1, x2, y2 = map(int, face['box'])
        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # display the frame in window
    cv.imshow("Video", frame)
    # Break the loop if 'q' is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the window
video.release()
cv.destroyAllWindows()
