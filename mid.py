import cv2
import numpy as np

#function to extract the object in the image
def extract(i):
    #convert the image to grayscale
    g = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
    #using threshold to get the edge of the object
    _, thresh_gray = cv2.threshold(g, thresh=1, maxval=255, type=cv2.THRESH_BINARY)
    image, contours,_ = cv2.findContours(thresh_gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    matrix = (0,0,0,0)
    matrix_area = 0
    #to find the object with the biggest bounding box
    for cont in contours:
        x,y,width,height = cv2.boundingRect(cont)
        area = width*height
        if area > matrix_area:
            matrix = x,y,width,height
            matrix_area = area
    x,y,width,height = matrix

    roi = i[y:y+height, x:x+width]
    #output of the cropped image
    cv2.imwrite('image2.png', roi)
    return roi

#helper function to calculate the thresh value
def thresh_help(img, lowThresholdRatio=0.05, highThresholdRatio=0.09):
    highThreshold = img.max() * highThresholdRatio;
    lowThreshold = highThreshold * lowThresholdRatio;
    M, N = img.shape
    res = np.zeros((M,N), dtype=np.int32)
    weak = np.int32(25)
    return weak

#main function
#def main():
#    img = cv2.imread('Detection.jpg')
    #gray = cv2.imread('Detection.png', cv2.IMREAD_GRAYSCALE)
#    new = extract(img)
#    cv2.imshow('new', new)
#    cv2.imshow('old', img)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
#main()
