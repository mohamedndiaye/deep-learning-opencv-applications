# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 16:34:12 2018

@author: rein9
"""

#facial landmark detection

#HOG + SVM
import numpy as np
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import dlib
import cv2
import argparse
import time

#VideoStream can catch the image of webcam with effcient fram pooling
#implemented using cv2.VideoCapture
# =============================================================================
# Arguments
# python facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg
# =============================================================================
ap = argparse.ArgumentParser()
ap.add_argument('-p', '--shape-predictor', required = True, help = 'Path to facial landmark predictor')
ap.add_argument('-r', '--piccamera', type = int, default = -1, help = 'whether or not the Rasberry Pi camera should be used')
ap.add_argument('-i', '--image', help='path to input image')
args = vars(ap.parse_args())

# =============================================================================
#dlib face detecotr, a HOG based facial landmark detector
# get_frontal_face_detector() -> dlib::object_detector<dlib::scan_fhog_pyramid<dlib::pyramid_down<6>,dlib::default_fhog_feature_extractor> >
# shape_predictor: loads a predictor and return a set of point locations that define the pose of the object
# =============================================================================
print("Loading Facila landmark predictor")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# =============================================================================
# Picture facial landmark detect
# =============================================================================
# detect faces in the grayscale image
image = cv2.imread(args["image"])
#image = imutils.resize(image, width=500)
IMG_SIZE = 500
image= cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
#need to normalize the image before sending it to predictor
print(image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

rects = detector(gray, 1)
# loop over the face detections
for (i, rect) in enumerate(rects):
  # determine the facial landmarks for the face region, then
  # convert the facial landmark (x, y)-coordinates to a NumPy
  # array
  shape = predictor(gray, rect)
  shape = face_utils.shape_to_np(shape)

# =============================================================================
# convert dlib's rectangle to a OpenCV-style bounding box
# [i.e., (x, y, w, h)], then draw the face bounding box
#
# rect.left()
# y = rect.top()
# w = rect.right() - x
# h = rect.bot() - y
# =============================================================================
  (x, y, w, h) = face_utils.rect_to_bb(rect)
  cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

  # show the face number
  cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

  # loop over the (x, y)-coordinates for the facial landmarks
  # and draw them on the image
  for (x, y) in shape:
    	cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

# show the output image with the face detections + facial landmarks
cv2.imshow("Output", image)
cv2.waitKey(0)


## =============================================================================
## For video pictures
## =============================================================================
## =============================================================================
## Initialize the Video Stream
## =============================================================================
#print("Warming up the webcam")
#vs = VideoStream(usePiCamera =args["picamera"] > 0).start()
#time.sleep(2.0)
#
## =============================================================================
##   grab the frame from teh threaded video stream, resize to have a max width of 400 pixels
## =============================================================================
#while True:
#  frame = vs.read()
#  (h,w) = frame.shape[:2]
#  frame = cv2.resize(frame, w, h, interpolation = cv2.INTER_AREA)
#  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#  #detect faces
#  rects = detector(gray, 0)
#
#for rect in rects:
#  #determine local facial landmark
#
#  shape= predictor(gray, rect)
#  shape= face_utils.shape_to_np(shape)#returns a list of (x,y) coordinates
#
#  for (x,y) in shape:
#    cv2.circle(frame, (x,y), 1, (0,0,255), -1)#draw a circle of on the output frame,
#
#  #show framme
#  cv2.imutils("Frame", frame)
#  key = cv2.waitKey(1) & 0xFF
#
#  #break from the loop if 'q' key was pressed
#  if key == ord('q'):
#    break
#cv2.destroyAllWindows()
#vs.stop()