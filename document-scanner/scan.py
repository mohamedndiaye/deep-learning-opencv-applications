# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 19:10:59 2018

@author: rein9
"""
# =============================================================================
# python scan.py --image images/page.jpg
# =============================================================================
from skimage import filters
import numpy as np
import argparse
import cv2
import imutils

# prepare arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required = True,
                 help = 'Path to the image to be scanned')
args= vars(ap.parse_args())

#load image and compute the h/w ratio and stretch to the new height
image = cv2.imread(args['image'])
ratio = image.shape[0]/500.0
orig = image.copy()
image = imutils.resize(image, height = 500)
# =============================================================================
#convert the image to grascale and blur it, to find the edges in the image
# =============================================================================
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#gausian blurr, to apply a gaussian filter to convolve the source image withe a specific Gaussian Kernel
# GaussianBlur(src, ksize, sigmaX[, dst[, sigmaY[, borderType]]]) -> dst
# Othre choices: filter2D, boxFilter, bilateralFilter, medianBlur
gray = cv2.GaussianBlur(gray, (5,5), 0)# 5 is kernel size , 0 for the standard deviation over X or Y direction
# Question: Canny vs HOG?
edged = cv2.Canny(gray, 75, 200)

#show the original image and the detection box
print('Edge Detection')
cv2.imshow('image',image)
cv2.imshow('edged', edged)
cv2.waitKey(0)
cv2.destroyAllWindows()
# =============================================================================
# Finding Contours
# find the contours in the edged image, keeping only the
# largest ones, and initialize the screen contour
# https://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html
# =============================================================================
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
print("Countour Dimension: ", len(cnts),"Coutour size %02d X %02d" % (len(cnts[0]), len(cnts[1])))
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

for c in cnts:
  peri = cv2.arcLength(c, True)  #Second argument specify whether shape is a closed contour (if passed True), or just a curve.
  approx = cv2.approxPolyDP(c, 0.02*peri, True)#approximates a contour shape to another shape with less number of vertices depending upon the precision we specify. It is an implementation of Douglas-Peucker algorithm.

  if len(approx) == 4:
    screenCnt = approx
    break #approximation successful
print("Find Contours of paper")
cv2.drawContours(image, [screenCnt], -1, (0,255,0), 2)#green color
cv2.imshow("Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
# =============================================================================
# Step 3: Apply a Perspective Transform & Threshold
# Threshold_local : Compute a threshold mask image based on local pixel neighborhood
#http://scikit-image.org/docs/dev/api/skimage.filters.htm
# =============================================================================
#apply the trnsformation to obtain a top/down view of the original image
from transform import four_point_transform
warped = four_point_transform(orig, screenCnt.reshape(4,2)*ratio)

#threshold it to give the black and white effect
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
print("Warped", warped)
T = filters.threshold_local(warped, 11, offset = 10, method = 'gaussian')
print("After threshold filtering", T)
warped = (warped > T).astype('uint8')*255

#show the original and scanned image
print("Apply perspective transform")
cv2.imshow("Original", imutils.resize(orig, height = 650))
cv2.imshow("Scanned", imutils.resize(warped, height = 650))
cv2.waitKey(0)
