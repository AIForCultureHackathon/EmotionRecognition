# -*- coding: utf-8 -*-
#

import face_recognition
import cv2
import argparse


# Arguments
parser = argparse.ArgumentParser(u"Face detection")
parser.add_argument("--movie", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
args = parser.parse_args()

# Input movie
input_movie = cv2.VideoCapture(args.movie)

# How many frames
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

# Create an output movie file
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_movie = cv2.VideoWriter('output.avi', fourcc, 25, (596, 336))

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
frame_number = 0

while True:
    # Grab a single frame of video
    ret, frame = input_movie.read()
    frame_number += 1

    # Quit when the input video file ends
    if not ret:
        break
    # end if

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)

    # Label the results
    for (top, right, bottom, left) in face_locations:
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
    # end for

    # Write the resulting image to the output video file
    print("Writing frame {} / {}".format(frame_number, length))
    output_movie.write(frame)
# end while

# All done!
input_movie.release()
cv2.destroyAllWindows()
