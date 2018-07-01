# -*- coding: utf-8 -*-
#

import face_recognition
import cv2
import argparse
from models import EmotionClassifier
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch


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
parser = argparse.ArgumentParser(u"Face detection")
parser.add_argument("--movie", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
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

# Create model
model = EmotionClassifier()

# Load model
model.load_state_dict(torch.load(open(args.model, 'rb')))

# Transformation to tensor and normalization
transform = transforms.Compose(
    [transforms.ToTensor()
     #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

# List of probs
probs_timeline = list()

# For each frame
while True:
    # Grab a single frame of video
    ret, frame = input_movie.read()

    # Quit when the input video file ends
    if not ret:
        break
    # end if

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)

    # Average probs
    average_probs = torch.zeros(1, 8)

    # How many faces
    n_faces = 0.0

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

        # Check size
        if face_image.shape[0] > 0 and face_image.shape[1] > 0:
            # Resize to 150x150
            face_image = cv2.resize(face_image, (150, 150))

            # Grayscale
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

            # Shape
            face_image.shape = (150, 150, 1)

            # Transform
            face_tensor = transform(face_image)

            # Add batch dim
            face_tensor = face_tensor.view(1, 1, 150, 150)

            # Model outputs
            outputs = model(Variable(face_tensor))

            # Emotion probs
            emotion_probs = torch.exp(outputs.data)
            average_probs += emotion_probs

            # Predict emotion
            _, predicted = torch.max(outputs.data, 1)
            predicted_emotion = classes[predicted[0]]

            # Draw a box around the face
            cv2.circle(frame, face_center, 5, (0, 0, 255))
            cv2.rectangle(
                frame,
                (face_left, face_top),
                (face_right, face_bottom),
                colors[predicted_emotion],
                4
            )

            # Plus one face
            n_faces += 1.0
        # end if
    # end for

    # Average probs
    average_probs /= n_faces

    # Add to timeline
    probs_timeline.append(average_probs)

    # Draw emotion timeline
    x = args.width - 1
    for t in range(len(probs_timeline)-1, 0, -1):
        probs_t = probs_timeline[t]
        # For each emotion
        for c in range(0, 8):
            em = classes[c]
            em_prob = probs_t[0, c]
            y = args.height - em_prob * 20.0 - 1.0
            if x > 0:
                print(x)
                print(y)
                cv2.circle(frame, (x, y), 2, colors[em], thickness=1)
            # end if
        # end for
        x -= 1
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
