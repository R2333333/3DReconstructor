import numpy as np

##@param verticalCenter: center berween widest points on vertical axis
##@param horizontalCenter: center berween widest points on horizontal axis
def mapping(front, left, top, verticalCenterL, verticalCenterF, horizontalCenter, horizontalCenterF):
    result = [] 
    result.extend(mapFrontBack(front, left, top, verticalCenterL, horizontalCenter))
    result.extend(mapTopBottom(front, left, top, verticalCenterL, horizontalCenterF))
    result.extend(mapLeftRight(front, left, top, verticalCenterL, verticalCenterF, horizontalCenter))

    return result

def mapFrontBack(front, left, top, verticalCenter, horizontalCenter):
    collect = []
    for row in range(front.shape[0]):
        if np.sum(front[row]) == 0:
            continue
        leftMost = nonzerofromLeft(left, row)
        rightMost = nonzerofromRight(left, row)
        for col in range(front.shape[1]):
            if front[row][col] != 0:
                frontMost = nonzerofromFront(top, col)
                if abs(rightMost - verticalCenter) < abs(frontMost - horizontalCenter):
                    collect.append((col, left.shape[1] - rightMost, front.shape[0] - row))
                else:
                    collect.append((col, left.shape[1] - frontMost, front.shape[0] - row))
                
                backMost = nonzerofromBack(top, col)
                if abs(leftMost - verticalCenter) < abs(backMost - horizontalCenter):
                    collect.append((col, left.shape[1] - leftMost, front.shape[0] - row))
                else:
                    collect.append((col, left.shape[1] - backMost, front.shape[0] - row))
    return collect

def mapLeftRight(front, left, top, verticalCenterL, verticalCenterF, horizontalCenterT):
    collect = []
    for row in range(left.shape[0]):
        if np.sum(left[row]) == 0:
            continue
        leftMost = nonzerofromLeft(front, row)
        rightMost = nonzerofromRight(front, row)
        for col in range(left.shape[1]):
            if left[row][col] != 0:
                frontMost = nonzerofromLeft(top, col)
                if abs(leftMost - verticalCenterF) < abs(frontMost - verticalCenterF):
                    collect.append((leftMost, left.shape[1]-col, left.shape[0] - row))
                else:
                    collect.append((frontMost, left.shape[1]-col, left.shape[0] - row))
                
                backMost = nonzerofromRight(top, col)
                if abs(rightMost - verticalCenterF) < abs(backMost - verticalCenterF):
                    collect.append((rightMost, left.shape[1]-col, left.shape[0] - row))
                else:
                    collect.append((backMost, left.shape[1]-col, left.shape[0] - row))
    return collect

def mapTopBottom(front, left, top, verticalCenter, horizontalCenter):
    collect = []
    for row in range(top.shape[0]):
        if np.sum(front[row]) == 0:
            continue
        downMost = nonzerofromFront(left, row)
        upMost = nonzerofromBack(left, row)
        for col in range(top.shape[1]):
            if top[row][col] != 0:
                frontMost = nonzerofromBack(front, col)
                if abs(upMost - horizontalCenter) < abs(frontMost - horizontalCenter):
                    collect.append((col, row, left.shape[0] - upMost))
                else:
                    collect.append((col, row, left.shape[0] - frontMost))
                
                backMost = nonzerofromFront(front, col)
                if abs(downMost - horizontalCenter) < abs(backMost - horizontalCenter):
                    collect.append((col, row, left.shape[0] - downMost))
                else:
                    collect.append((col, row, left.shape[0] - backMost))
    return collect



#used for left image
def nonzerofromLeft(image, z):
    for i in range(0, image.shape[1]):
        if image[z][i] != 0:
            return i

def nonzerofromRight(image, z):
    for i in range(image.shape[1]-1, -1, -1):
        if image[z][i] != 0:
            return i

#used for top image
def nonzerofromBack(image, x):
    for i in range(0, image.shape[0]):
        if image[i][x] != 0:
            return i


def nonzerofromFront(image, x):
    for i in range(image.shape[0]-1, -1, -1):
        # print(i, ', ', x)
        if image[i][x] != 0:
            return i


