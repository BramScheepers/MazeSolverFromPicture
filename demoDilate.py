from customDilate import dilate
import cv2 as cv
import numpy as np
image = cv.imread('dilate.png')
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
r, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
dilated = dilate(~thresh, 3, 10)
dilate2 = cv.dilate(~thresh, np.ones((3,3), np.uint8), iterations=10)
cv.imshow('tsdf', dilated)
cv.imshow('2', dilate2)
cv.waitKey(0)
cv.destroyAllWindows()