import numpy as np

##@param verticalCenter: center berween widest points on vertical axis
##@param horizontalCenter: center berween widest points on horizontal axis
def mapping(front, left, top, verticalCenter, horizontalCenter):
    result, l, r = [], 0, 0 # l is the left point on the edge, same with r

    for z in range(0, front.shape[0]):
        if np.sum(front[z]) == 0:
            continue
        for x in range(0, front.shape[1]):
            if x == 0:
                continue

