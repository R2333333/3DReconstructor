import cv2
import numpy as np

#function to extract the only object in the image
def extract(i):
    g = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
    retval, thresh_gray = cv2.threshold(g, thresh=100, maxval=255, type=cv2.THRESH_BINARY_INV)
    image, contours, hierarchy = cv2.findContours(thresh_gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    mx = (0,0,0,0)
    mx_area = 0
    for cont in contours:
        x,y,w,h = cv2.boundingRect(cont)
        area = w*h
        if area > mx_area:
            mx = x,y,w,h
            mx_area = area
    x,y,w,h = mx

    roi = i[y:y+h, x:x+w]
    cv2.imwrite('image2.png', roi)
    return roi

#main function
def main():
    img = cv2.imread('image.png')
    gray = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)
    new = extract(img)
    cv2.imshow('new', new)
    cv2.imshow('old', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
main()
