import cv2
import numpy as np

#function to extract the object in the image
def extract(i):
    #convert the image to grayscale
    #g = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
    i = i.astype(np.uint8)
    #using threshold to get the edge of the object
    _, thresh_gray = cv2.threshold(i, thresh=1, maxval=255, type=cv2.THRESH_BINARY)
    _,contours,_ = cv2.findContours(thresh_gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

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

#put img1 into the center of the img2
def merge(img1, img2):
    #the shape of the inserted image
    h, w = img1.shape
    print(h,w)
    #thje shape of the background image
    hh, ww = img2.shape
    print(hh,ww)

    # compute xoff and yoff for placement of upper left corner of resized image
    yoff = round((hh-h)/2)
    xoff = round((ww-w)/2)
    print(yoff,xoff)

    # use numpy indexing to place the resized image in the center of background image
    result = img2.copy()
    result[yoff:yoff+h, xoff:xoff+w] = img1

    # view result
    cv2.imshow('CENTERED', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # save resulting centered image
    cv2.imwrite('center.png', result)
    return result

#calculate the center of the iput image.
def get_center(img):
    x = img.shape[0]
    y = img.shape[1]
    x_c = 0
    y_c = 0
    if (x%2 == 0):
        x_c = x/2-1
    else:
        x_c = x/2 + 0.5
    if (y%2 == 0):
        y_c = y/2-1
    else:
        y_c = y/2 + 0.5

    print(x_c, y_c)
