# -*- coding: utf-8 -*-
#

import face_recognition
import cv2
import argparse
import os


# Arguments
parser = argparse.ArgumentParser(u"Face detection")
parser.add_argument("--movie", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--faces-dir", type=str, required=True)
parser.add_argument("--model", type=str, required=False)
parser.add_argument("--length", type=int, default=-1)
parser.add_argument("--fps", type=int, required=True)
parser.add_argument("--width", type=int, required=True)
parser.add_argument("--height", type=int, required=True)
args = parser.parse_args()

# Input movie
input_movie = cv2.VideoCapture(args.movie)

# How many frames
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

# Create an output movie file
fourcc = cv2.VideoWriter_fourcc(*'mpeg')
output_movie = cv2.VideoWriter(args.output, fourcc, args.fps, (args.width, args.height))

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
frame_number = 0

# Index
index = 0

# For each frame
while True:
    # Grab a single frame of video
    ret, frame = input_movie.read()

    # Quit when the input video file ends
    """if not ret:
        break
    # end if"""

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)

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

        # Draw a box around the face
        cv2.circle(frame, face_center, 5, (0, 0, 255))
        # cv2.circle(frame, (left, top), 10, (0, 0, 255))
        cv2.rectangle(
            frame,
            (int(face_center[0] - biggest_dim / 2.0), int(face_center[1] - biggest_dim / 2.0)),
            (int(face_center[0] + biggest_dim / 2.0), int(face_center[1] + biggest_dim / 2.0)),
            (0, 0, 255),
            2
        )

        # Face
        # face_image = frame[top:bottom, left:right]

        # Save image
        # cv2.imwrite(os.path.join(args.faces_dir, "face" + str(index) + ".jpg"), face_image)

        # Next face
        # index += 1
    # end for

    # Write the resulting image to the output video file
    print("Writing frame {} / {}".format(frame_number, length))
    output_movie.write(frame)

    # Frame number
    frame_number += 1

    # Check length
    if args.length != -1 and frame_number > args.length:
        print(u"The end...")
        break
    # end if
# end while

# All done!
input_movie.release()
output_movie.release()
cv2.destroyAllWindows()
