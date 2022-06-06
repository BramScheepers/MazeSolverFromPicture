from customFunctions import dilate
from customFunctions import erode
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def resize_image_and_show(image):
    preRezised = image
    scale_percent = 20
    # width = int(preRezised.shape[1] * scale_percent / 100)
    # height = int(preRezised.shape[0] * scale_percent / 100)
    width = int(1000)
    height = int(1000)
    dim = (width, height)
    Resized = cv.resize(preRezised, dim, interpolation=cv.INTER_AREA)

    # Show scaled image
    cv.imshow("Resized", Resized)
    # return Resized
    cv.waitKey(0)
    cv.destroyAllWindows()

image = cv.imread('dilatedemo.png')
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
r, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

plt.subplot(221), plt.imshow(~thresh, cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(erode(~thresh, 3, 1), cmap='gray')
plt.title('Eroded 3x3 1 iteration'), plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(erode(~thresh, 3, 4), cmap='gray')
plt.title('Eroded 3x3 4 iterations'), plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(erode(~thresh, 7, 1), cmap='gray')
plt.title('Eroded 7x7 1 iteration'), plt.xticks([]), plt.yticks([])


plt.show()
