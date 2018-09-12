# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 14:09:31 2018

@author: rein9
"""

import cv2
import numpy as np

def order_points(pts):
  #order points so that 1st point is tl, tr, br, bl
  rect = np.zeros((4,2), dtype= np.float32)
  #tl has the smallest sum, because of the way array indexes
  s = pts.sum(axis = 1)
  rect[0] = pts[np.argmin(s)] #tl
  rect[2] = pts[np.argmax(s)]

  diff = np.diff(pts, axis =1) # axis = 1 gives the difference along the same row
  rect[1] = pts[np.argmin(diff)] #tr
  rect[3] = pts[np.argmax(diff)]

  return rect

def four_point_transform(img, pts):
  rect = order_points(pts)
  (tl,tr,br,bl) = rect
  #get the width
  widthA = np.sqrt((br[0]-bl[0])**2 + (br[1]-bl[1])**2)
  widthB = np.sqrt((tr[0]-tl[0])**2 + (tr[1]-tl[1])**2)
  maxWidth = max(int(widthA), int(widthB))
  #get the height
  heightA = np.sqrt((br[0]-tr[0])**2 + (br[1]-tr[1])**2)
  heightB = np.sqrt((bl[0]-tl[0])**2 + (bl[1]-tl[1])**2)
  maxHeight = max(int(heightA), int(heightB))

  #construct the destination points, tl,tr, br, bl
  dst = np.array([
      [0,0],
      [maxWidth-1,0],
      [maxWidth-1, maxHeight-1],
      [0, maxHeight-1]], dtype = "float32")
  #get the transformation
  M = cv2.getPerspectiveTransform(rect, dst)
  warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

  return warped