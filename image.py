import numpy as np
import sys
from matplotlib import pyplot as plt
# np.set_printoptions(threshold=sys.maxsize)

def drawSame():
    front, left, top = np.zeros((200,200)), np.zeros((200,200)), np.zeros((200,200))
    front[25:175, 50:150] = (front + 1)[25:175, 50:150]
    left[50:150, 50:150] = (left + 1)[50:150, 50:150]
    top += front

    return front, left, top

front, left, top = drawSame()

# print(front[25:50, 40:50])

# plt.imshow(front)
# plt.show()
# plt.imshow(left)
# plt.show()
# plt.imshow(top)
# plt.show()

