# -*- coding: utf-8 -*-
#

# Imports
import cv2
import sys


# Classifier path
class_path = "."
face_cascade = cv2.CascadeClassifier(class_path)

# Get video devices
video_capture = cv2.VideoCapture(0)

# Capture frame by frame
while True:
    # Get the frame
    _, frame = video_capture.read()

    # To gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Classifier
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CV_HAAR_SCALE_IMAGE
    )

    # Draw rectangle around each faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # end for

    # Display the frame
    cv2.imshow('Video', frame)

    # If exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # end if
# end while

# Release
video_capture.release()
cv2.destroyAllWindows()

