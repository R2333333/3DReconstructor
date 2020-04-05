import cv2
import numpy as np
from edge import gaussian_filter
from edge import non_max_suppression
from edge import convolution
from edge import sobel

img = cv2.imread("img.jpg", cv2.IMREAD_GRAYSCALE)
gray = cv2.GaussianBlur(img, (5,5), 0)
g, d = sobel(gray)
nms = non_max_suppression(g, d)

#document to save pixel for checking
def doc(image):
    with open("123.txt", 'w') as f:
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                f.write(str(image[x,y]) + " ")
            f.write("\n")
    f.close()

def main():
    final = np.empty(nms.shape, dtype = np.float32)
    for i in range(nms.shape[0]):
        for j in range(nms.shape[1]):
            if nms[i,j] > 0:
                final[i,j] = nms[i,j]

    doc(final)
    cv2.imwrite("final.jpg", final)
    cv2.imshow("final", final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
main()
