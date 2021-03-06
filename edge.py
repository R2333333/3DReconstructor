import numpy as np
import cv2
from image import drawDiff
import itertools
from matplotlib import pyplot as plt


# gaussian filter
def gaussian_filter(sigma):
    size = 2 * np.ceil(3 * sigma) + 1
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) / (2 * np.pi * sigma ** 2)
    return g / g.sum()


# non maximum suooression
def non_max_suppression(img, direction):
    X, Y = img.shape
    Z = np.zeros((X, Y), dtype=np.int32)
    angle = direction * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, X - 1):
        for j in range(1, Y - 1):
            try:
                q = 255
                p = 255
                # angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = img[i, j - 1]
                    p = img[i, j + 1]
                # angle 45
                elif (22.5 <= angle[i, j] < 67.5):
                    q = img[i + 1, j - 1]
                    p = img[i - 1, j + 1]
                # amgle 90
                elif (67.5 <= angle[i, j] < 112.5):
                    q = img[i + 1, j]
                    p = img[i - 1, j]
                # angle 135
                elif (112.5 <= angle[i, j] < 157.5):
                    q = img[i - 1, j - 1]
                    p = img[i + 1, j + 1]

                if (img[i, j] >= q) and (img[i, j] >= p):
                    Z[i, j] = img[i, j]
                else:
                    Z[i, j] = 0
            except IndexError as e:
                pass
    return Z


# sobel filter
def sobel(img, convert=False):
    # x-oriented Sobel fiter
    sobelX = np.array((
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]), np.float32)
    # y-oriented Sobel fiter
    sobelY = np.array((
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]), np.float32)
    Ix = convolution(img, sobelX)
    Iy = convolution(img, sobelY)
    # cv2.imwrite("X.jpg", Ix)
    # cv2.imwrite("Y.jpg", Iy)
    # magnitude
    G = np.sqrt(np.square(Ix) + np.square(Iy))
    G = G / np.max(G) * 255
    # direction
    direction = np.arctan2(Iy, Ix)
    return (G, direction.astype(np.float32))


# image filtering
def convolution(img, kernel, average=False):
    # get image width and height
    image_row, image_col = img.shape
    # get kernel's width and height
    kernel_row, kernel_col = kernel.shape
    # output image
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


def get_pixel_set(img):
    pixel_set = []
    X, Y = img.shape
    left_border = 0
    right_border = 0
    top_border = 0
    bottom_border = 0
    for x in range(0, X):
        for y in range(0, Y):
            if img[x, y] > 0:
                pixel_set.append([x, y])

    for x in range(0, X):
        for y in range(0, Y):
            if img[x, y] > 0:
                left_border = x
                break

    for x in range(X - 1, -1, -1):
        for y in range(Y - 1, -1, -1):
            if img[x, y] > 0:
                right_border = x
                break

    for y in range(0, Y):
        for x in range(0, X):
            if img[x, y] > 0:
                top_border = y
                break

    for y in range(Y - 1, -1, -1):
        for x in range(X - 1, -1, -1):
            if img[x, y] > 0:
                bottom_border = y
                break

    y_axial_length = left_border - right_border
    x_axial_length = top_border - bottom_border

    return pixel_set, y_axial_length, x_axial_length


def find_scale_image(fxl, fyl, lxl, lyl, txl, tyl):
    if fxl / txl < 1 and fyl / lyl < 1:
        # resize to front scale
        index = 1
    elif fxl / txl > 1 and fyl / lyl < 1:
        # resize to top scale
        index = 2
    elif fxl / txl < 1 and fyl / lyl > 1:
        # resize to left scale
        index = 3
    else:
        if tyl / lxl > 1:
            # resize to left scale
            index = 3
        else:
            # resize to top scale
            index = 2
    return index


def resize(img1, img2, img3, r1, r2):
    Xf, Yf = img1.shape
    Xl, Yl = img2.shape
    Xt, Yt = img3.shape
    # ratio for left img
    ratio1 = r1
    resize_left_x = int(Xl * ratio1)
    resize_left_y = int(Yl * ratio1)
    # print(resize_left_x)
    # print(resize_left_y)
    dim_left = (resize_left_x, resize_left_y)
    resize_left = cv2.resize(img2, dim_left)
    # ratio for top img
    ratio2 = r2
    resize_top_x = int(Xt * ratio2)
    resize_top_y = int(Yt * ratio2)
    # print(resize_top_x)
    # print(resize_top_y)
    dim_top = (resize_top_x, resize_top_y)
    resize_top = cv2.resize(img3, dim_top)
    # border = (kX - 1) // 2
    # image = cv2.copyMakeBorder(image, border, border, border, border, cv2.BORDER_CONSTANT)
    resize_Xl, resize_Yl = resize_left.shape
    resize_Xt, resize_Yt = resize_top.shape
    # top_bottom_Lborder = (Xf - resize_Xl) // 2
    # left_right_Lborder = (Yf - resize_Yl) // 2
    # top_bottom_Tborder = (Xf - resize_Xt) // 2
    # left_right_Tborder = (Yf - resize_Yt) // 2
    top_Lborder, bottom_Lborder = make_border(Xf, resize_Xl)
    left_Lborder, right_Lborder = make_border(Yf, resize_Yl)
    top_Tborder, bottom_Tborder = make_border(Xf, resize_Xt)
    left_Tborder, right_Tborder = make_border(Yf, resize_Yt)
    fullup_left = cv2.copyMakeBorder(resize_left, top_Lborder, bottom_Lborder, left_Lborder, right_Lborder, cv2.BORDER_REPLICATE)
    fullup_top = cv2.copyMakeBorder(resize_top, top_Tborder, bottom_Tborder, left_Tborder, right_Tborder, cv2.BORDER_REPLICATE)
    # fXl, fYl = fullup_left.shape
    # fXt, fYt = fullup_top.shape
    # print(fXl)
    # print(fYl)
    # print(fXt)
    # print(fYt)
    return resize_left, resize_top


def make_border(large, small):
    if (large - small) % 2 == 0:
        top_border = (large - small) // 2
        bottom_border = (large - small) // 2
    else:
        top_border = ((large - small) - 1) // 2
        bottom_border = ((large - small) + 1) // 2

    return top_border, bottom_border


def return_3_view_img():
    # main function
    front, left, top = drawDiff()
    gray_front = front
    gray_left = left
    gray_top = top
    new_kernel = gaussian_filter(1)
    newImg_front = convolution(gray_front, new_kernel)
    newImg_left = convolution(gray_left, new_kernel)
    newImg_top = convolution(gray_top, new_kernel)
    gf, df = sobel(newImg_front)
    gl, dl = sobel(newImg_left)
    gt, dt = sobel(newImg_top)
    final_front = non_max_suppression(gf, df)
    final_left = non_max_suppression(gl, dl)
    final_top = non_max_suppression(gt, dt)

    # now we put every pixels with color into a set for three view images, use these sets to rescale our object later
    front_with_color_pixel, front_y_length, front_x_length = get_pixel_set(gray_front)
    left_with_color_pixel, left_y_length, left_x_length = get_pixel_set(gray_left)
    top_with_color_pixel, top_y_length, top_x_length = get_pixel_set(gray_top)
   
    check_num = find_scale_image(front_x_length, front_y_length, left_x_length, left_y_length, top_x_length,
                                 top_y_length)
    # print(check_num)

    if check_num == 1:
        # resize left and top imgs with front img scale
        r1 = front_y_length / left_y_length
        r2 = front_x_length / top_x_length
        fullup_img1, fullup_img2 = resize(gray_front, gray_left, gray_top, r1, r2)
    elif check_num == 2:
        # resize left and front imgs with top img scale
        r1 = top_y_length / left_x_length
        r2 = top_x_length / front_x_length
        fullup_img1, fullup_img2 = resize(gray_top, gray_left, gray_front, r1, r2)
    elif check_num == 3:
        # resize top and front imgs with left img scale
        r1 = left_x_length / top_y_length
        r2 = left_y_length / front_y_length
        fullup_img1, fullup_img2 = resize(gray_left, gray_top, gray_front, r1, r2)

    newImg_resize_left = convolution(fullup_img1, new_kernel)
    newImg_resize_top = convolution(fullup_img2, new_kernel)
    grl, drl = sobel(newImg_resize_left)
    grt, drt = sobel(newImg_resize_top)
    final_resize_img1 = non_max_suppression(grl, drl)
    final_resize_img2 = non_max_suppression(grt, drt)

    titles = ['front', 'left', 'top', 'resize img1', 'resize img2', 'final img1', 'final img2', 'final top', 'fr1',
              'fr2']
    images = [front, left, top, fullup_img1, fullup_img2, final_front, final_left, final_top, final_resize_img1,
              final_resize_img2]

    _, img1y, img1x = get_pixel_set(gray_top)
    _, img2y, img2x = get_pixel_set(fullup_img1)
    _, img3y, img3x = get_pixel_set(fullup_img2)

    for i in range(10):
        plt.subplot(2, 5, i + 1), plt.imshow(images[i], 'gray'), plt.axis(sharex=True, sharey=True)
        plt.title(titles[i])

    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if check_num == 1:
        # return front, left, top
        return gray_front, fullup_img1, fullup_img2
    elif check_num == 2:
        return fullup_img2, fullup_img1, gray_top
    elif check_num == 3:
        return fullup_img2, gray_left, fullup_img1
