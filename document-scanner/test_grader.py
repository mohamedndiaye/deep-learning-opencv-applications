# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 15:02:13 2018

@author: rein9
"""
# =============================================================================
#python test_grader.py --image images/test_01.png
#Step #1: Detect the exam in an image.
#Step #2: Apply a perspective transform to extract the top-down, birds-eye-view of the exam.
#Step #3: Extract the set of bubbles (i.e., the possible answer choices) from the perspective transformed exam.
#Step #4: Sort the questions/bubbles into rows.
#Step #5: Determine the marked (i.e., “bubbled in”) answer for each row.
#Step #6: Lookup the correct answer in our answer key to determine if the user was correct in their choice.
#Step #7: Repeat for all questions in the exam.
# =============================================================================
# import the necessary packages
from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
args = vars(ap.parse_args())

# define the answer key which maps the question number
# to the correct answer
ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}

# load the image, convert it to grayscale, blur it
# slightly, then find edges
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 75, 200)

#fins the contour map, then initialize the contour that correspnds the documents
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
docCnt = True

if len(cnts)>0:
  #sort the contours according to their seize in descending order
  cnts = sorted(cnts, key=cv2.contourArea, reverse = True)
  for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02*peri, True)

    if len(approx) ==4:
      # if the approximation has 4 points, we can assume we found the paper
      docCnt = approx
      break

# =============================================================================
#   #apply four point transformation
# =============================================================================
paper = four_point_transform(image, docCnt.reshape(4,2))
warped = four_point_transform(gray, docCnt.reshape(4,2))

  #WATCH OUT HERE, we are not using thresh_local any more, we are using thresh_OST
  #apply OTSU filter to binarize the warped piece of the paper
thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
#after thresholding, Notice how the background of the image is black, while the foreground is white.

# =============================================================================
# Need to contour again on the binarized image
# =============================================================================
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
questionCnts = []

for c in cnts:
  # compute the bounding box of the contour, then use the
  # bounding box to derive the aspect ratio
  (x,y,w,h) = cv2.boundingRect(c)
  ar = w/float(h)

  	# in order to label the contour as a question, region
	# should be sufficiently wide, sufficiently tall, and
	# have an aspect ratio approximately equal to 1
  if w >=20 and h >=20 and ar >=0.9 and ar<=1.1:
    questionCnts.append(c)

# sort the question contours top-to-bottom, then initialize
# the total number of correct answers
questionCnts = contours.sort_contours(questionCnts, method = "top-to-bottom")[0]
correct = 0

#loop over the see which one of the 5 possible answers is selected
for (q,i) in enumerate(np.arange(0, len(questionCnts), 5)):
  cnts = contours.sort_contours(questionCnts[i:i+5])[0]
  bubbled = None

  for (j,c) in enumerate(cnts):
    #construct a mask that reveals only the current "bubble" for the question
    mask = np.zeros(thresh.shape, dtype= 'uint8')
    cv2.drawContours(mask, [c], -1, 255, -1)
    #apple the mask to the thresholded image, then count the number of non-zero pixels in the bubble area
    mask = cv2.bitwise_and(thresh, thresh, mask=mask)
    total = cv2.countNonZero(mask)
    # if the current total has a larger number of total
		# non-zero pixels, then we are examining the currently
		# bubbled-in answer
    if bubbled is None or total > bubbled[0]:
      bubbled = (total, j)

  color = (0,0,255)
  k = ANSWER_KEY[q]
  #check to see if the bubbled answer if correct
  if k == bubbled[1]:
    color= (0,255,0)
    correct+=1
  # draw the outline of the correct answer
  cv2.drawContours(paper, [cnts[k]], -1, color, 3)


# grab the test taker
score = (correct / 5.0) * 100
print("[INFO] score: {:.2f}%".format(score))
cv2.putText(paper, "{:.2f}%".format(score), (10, 30),
	cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
cv2.imshow("Original", image)
cv2.imshow("Exam", paper)
cv2.waitKey(0)