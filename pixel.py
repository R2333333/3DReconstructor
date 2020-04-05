import cv2
import numpy as np
from edge import gaussian_filter
from edge import non_max_suppression
from edge import convolution
from edge import sobel
from image import drawSame
import sys
from matplotlib import pyplot as plt

#img = cv2.imread("img.jpg", cv2.IMREAD_GRAYSCALE)
front, left, top = drawSame()
gray = cv2.GaussianBlur(front, (5,5), 0)
g, d = sobel(gray)
nms = non_max_suppression(g, d)

final = np.empty(nms.shape, dtype = np.float32)
for i in range(nms.shape[0]):
	for j in range(nms.shape[1]):
		if nms[i,j] > 0:
			final[i,j] = 1
		else:
			final[i,j] = 0
plt.imshow(final)
plt.show()




