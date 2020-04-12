import numpy as np
import sys
from matplotlib import pyplot as plt
# np.set_printoptions(threshold=sys.maxsize)

def drawSame():
    front, left, top = np.zeros((200,200)), np.zeros((200,200)), np.zeros((200,200))
    left[25:175, 50:150] = (left + 1)[25:175, 50:150]
    top[20:180, 20:180] = (top + 1)[20:180, 20:180]
    front[40:160, 60:140] = (front + 1)[40:160, 60:140]

    return front, left, top


front, left, top = drawSame()

# print(front[25:50, 40:50])

# plt.imshow(front)
# plt.show()
# plt.imshow(left)
# plt.show()
# plt.imshow(top)
# plt.show()

