import numpy as np
import cv2
from image import drawSame

#gaussian filter
def gaussian_filter(sigma):
    size = 2 * np.ceil(3*sigma) + 1
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2))) / (2*np.pi*sigma**2)
    return g/g.sum()
#non maximum suooression
def non_max_suppression(img, direction):
    X, Y = img.shape
    Z = np.zeros((X, Y), dtype = np.int32)
    angle = direction * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, X-1):
        for j in range(1, Y-1):
            try:
                q = 255
                p = 255
                #angle 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = img[i, j-1]
                    p = img[i, j+1]
                #angle 45
                elif (22.5 <= angle[i,j] < 67.5):
                    q = img[i+1, j-1]
                    p = img[i-1, j+1]
                #amgle 90
                elif (67.5 <= angle[i,j] < 112.5):
                    q = img[i+1, j]
                    p = img[i-1, j]
                #angle 135
                elif (112.5 <= angle[i,j] < 157.5):
                    q = img[i-1, j-1]
                    p = img[i+1, j+1]

                if (img[i,j] >= q) and (img[i,j] >= p):
                    Z[i,j] = img[i,j]
                else:
                    Z[i,j] = 0
            except IndexError as e:
                pass
    return Z
#sobel filter
def sobel(img, convert=False):
    #x-oriented Sobel fiter
    sobelX = np.array((
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]), np.float32)
    #y-oriented Sobel fiter
    sobelY = np.array((
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]), np.float32)
    Ix = convolution(img, sobelX)
    Iy = convolution(img, sobelY)
    #cv2.imwrite("X.jpg", Ix)
    #cv2.imwrite("Y.jpg", Iy)
    #magnitude
    G = np.sqrt(np.square(Ix) + np.square(Iy))
    G = G/np.max(G) * 255
    #direction
    direction = np.arctan2(Iy, Ix)
    return (G, direction.astype(np.float32))

#image filtering
def convolution(img, kernel, average=False):
    #get image width and height
    image_row, image_col = img.shape
    #get kernel's width and height
    kernel_row, kernel_col = kernel.shape
    #output image
    output = np.zeros(img.shape)
    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)
    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))
    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = img
    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
            if average:
                output[row, col] /= kernel.shape[0] * kernel.shape[1]
    return output

#main function
# gray = cv2.imread("img0.jpg", cv2.IMREAD_GRAYSCALE)
front, left, top = drawSame()
gray = front
new_kernel = gaussian_filter(1)
newImg = convolution(gray, new_kernel)
g, d = sobel(newImg)
final = non_max_suppression(g, d)
#cv2.imwrite("gradient orientation.jpg", d)
#cv2.imwrite("gradient magnitude.jpg", g)
cv2.imshow("Detection", final.astype("uint8"))
cv2.waitKey(0)
cv2.destroyAllWindows()
