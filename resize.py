# try to resize by hand, but the result is not accurate

# Xl, Yl = final_left.shape
# Xt, Yt = final_top.shape
# if check_num == 1:
#     Xl, Yl = final_left.shape
#     Xt, Yt = final_top.shape
#     # with the mid point of two larger images, we fix the size now
#     mid_Xl = Xl / 2
#     mid_Yl = Yl / 2
#     mid_Xt = Xt / 2
#     mid_Yt = Yt / 2
#     ratio1 = front_y_length / left_y_length
#     ratio2 = front_x_length / top_x_length
#     # fix length img size
#     for pixel in left_with_color_pixel:
#         if pixel[0] < mid_Xl and pixel[1] < mid_Yl:
#             pixel[0] = int(mid_Xl - (mid_Xl - pixel[0]) * ratio1)
#             pixel[1] = int(mid_Yl - (mid_Yl - pixel[1]) * ratio1)
#         elif pixel[0] > mid_Xl and pixel[1] < mid_Yl:
#             pixel[0] = int(mid_Xl + (pixel[0] - mid_Xl) * ratio1)
#             pixel[1] = int(mid_Yl - (mid_Yl - pixel[1]) * ratio1)
#         elif pixel[0] < mid_Xl and pixel[1] > mid_Yl:
#             pixel[0] = int(mid_Xl - (mid_Xl - pixel[0]) * ratio1)
#             pixel[1] = int(mid_Yl + (pixel[1] - mid_Yl) * ratio1)
#         elif pixel[0] > mid_Xl and pixel[1] > mid_Yl:
#             pixel[0] = int(mid_Xl + (pixel[0] - mid_Xl) * ratio1)
#             pixel[1] = int(mid_Yl + (pixel[1] - mid_Yl) * ratio1)
#         elif pixel[0] < mid_Xl and pixel[1] == mid_Yl:
#             pixel[0] = int(mid_Xl - (mid_Xl - pixel[0]) * ratio1)
#         elif pixel[0] > mid_Xl and pixel[1] == mid_Yl:
#             pixel[0] = int(mid_Xl + (pixel[0] - mid_Xl) * ratio1)
#         elif pixel[0] == mid_Xl and pixel[1] < mid_Yl:
#             pixel[1] = int(mid_Yl - (mid_Yl - pixel[1]) * ratio1)
#         elif pixel[0] == mid_Xl and pixel[1] > mid_Yl:
#             pixel[1] = int(mid_Yl + (pixel[1] - mid_Yl) * ratio1)
#
#     for pixel in top_with_color_pixel:
#         if pixel[0] < mid_Xt and pixel[1] < mid_Yt:
#             pixel[0] = int(mid_Xt - (mid_Xt - pixel[0]) * ratio2)
#             pixel[1] = int(mid_Yt - (mid_Yt - pixel[1]) * ratio2)
#         elif pixel[0] > mid_Xt and pixel[1] < mid_Yt:
#             pixel[0] = int(mid_Xt + (pixel[0] - mid_Xt) * ratio2)
#             pixel[1] = int(mid_Yt - (mid_Yt - pixel[1]) * ratio2)
#         elif pixel[0] < mid_Xt and pixel[1] > mid_Yt:
#             pixel[0] = int(mid_Xt - (mid_Xt - pixel[0]) * ratio2)
#             pixel[1] = int(mid_Yt + (pixel[1] - mid_Yt) * ratio2)
#         elif pixel[0] > mid_Xt and pixel[1] > mid_Yt:
#             pixel[0] = int(mid_Xt + (pixel[0] - mid_Xt) * ratio2)
#             pixel[1] = int(mid_Yt + (pixel[1] - mid_Yt) * ratio2)
#         elif pixel[0] < mid_Xt and pixel[1] == mid_Yt:
#             pixel[0] = int(mid_Xt - (mid_Xt - pixel[0]) * ratio2)
#         elif pixel[0] > mid_Xt and pixel[1] == mid_Yt:
#             pixel[0] = int(mid_Xt + (pixel[0] - mid_Xt) * ratio2)
#         elif pixel[0] == mid_Xt and pixel[1] < mid_Yt:
#             pixel[1] = int(mid_Yt - (mid_Yt - pixel[1]) * ratio2)
#         elif pixel[0] == mid_Xt and pixel[1] > mid_Yt:
#             pixel[1] = int(mid_Yt + (pixel[1] - mid_Yt) * ratio2)
# # print(left_with_color_pixel)
# # print(top_with_color_pixel)
# left_with_color_pixel.sort()
# new_left_set = list(left_with_color_pixel for left_with_color_pixel, _ in itertools.groupby(left_with_color_pixel))
# print(new_left_set)
#
# top_with_color_pixel.sort()
# new_top_set = list(top_with_color_pixel for top_with_color_pixel, _ in itertools.groupby(top_with_color_pixel))
# print(new_top_set)
#
# resize_left, resize_top = np.zeros((200,200)), np.zeros((200,200))
# Xrl, Yrl = resize_left.shape
# Xrt, Yrt = resize_top.shape
# for x in range(0, Xrl):
#     for y in range(0, Yrl):
#         for pixel in new_left_set:
#             if x == pixel[0] and y == pixel[1]:
#                 resize_left[x, y] = (resize_left + 1)[x, y]
#                 # new_left_set.remove(pixel)
#
# for x in range(0, Xrt):
#     for y in range(0, Yrt):
#         for pixel in new_top_set:
#             if x == pixel[0] and y == pixel[1]:
#                 resize_top[x, y] = (resize_top + 1)[x, y]
#                 # new_left_set.remove(pixel)
