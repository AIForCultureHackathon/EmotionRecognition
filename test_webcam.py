# -*- coding: utf-8 -*-
#

import cv2
import argparse
from models import EmotionClassifier
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch
import math
import face_recognition
import numpy as np


# Classes
classes = ('neutral', 'happiness', 'surprise', 'anger', 'sadness', 'disgust', 'fear', 'contempt')
colors = {
    'neutral': (255, 255, 255),
    'happiness': (0, 255, 0),
    'surprise': (0, 0, 255),
    'anger': (255, 0, 0),
    'sadness': (255, 255, 0),
    'disgust': (255, 0, 255),
    'fear': (0, 0, 0),
    'contempt': (0, 255, 255)
}


# Arguments
parser = argparse.ArgumentParser(u"Emotion Reaction Measurement Software")
parser.add_argument("--model", type=str, required=False)
args = parser.parse_args()

# Input movie
input_device = cv2.VideoCapture(0)

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []

# Frame index
frame_index = 0

# For each frame
while True:
    # Capture a frame
    ret, frame = input_device.read()

    if frame_index % 4 == 0:
        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = frame[:, :, ::-1]

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_frame)

        # Face landmarks
        face_landmarks_list = face_recognition.face_landmarks(rgb_frame)

        # For each landmarks
        for face_landmarks in face_landmarks_list:
            # For each part
            for part in face_landmarks.keys():
                pts = np.array(face_landmarks[part], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], True, (255, 255, 255))
            # end for
        # end for

        # Label the results
        for (top, right, bottom, left) in face_locations:
            # Face size
            face_size = (right - left, bottom - top)

            # Get the center of the face
            face_center = (int(left + face_size[0] / 2.0), int(top + face_size[0] / 2.0))

            # Biggest dim
            if face_size[0] > face_size[1]:
                biggest_dim = face_size[0]
            else:
                biggest_dim = face_size[1]
            # end if

            # Face rectangle
            face_top = int(face_center[1] - biggest_dim / 2.0)
            face_bottom = int(face_center[1] + biggest_dim / 2.0)
            face_left = int(face_center[0] - biggest_dim / 2.0)
            face_right = int(face_center[0] + biggest_dim / 2.0)

            # Face
            face_image = frame[face_top:face_bottom, face_left:face_right]

            # Draw a box around the face
            cv2.circle(frame, face_center, 5, (0, 0, 255))
            cv2.rectangle(
                frame,
                (face_left, face_top),
                (face_right, face_bottom),
                (255, 255, 255),
                4
            )
        # end for

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # end for
    # end if

    # Frame index
    frame_index += 1
# end while

# All done!
input_device.release()
cv2.destroyAllWindows()
