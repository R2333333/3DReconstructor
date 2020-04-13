import numpy as np
import sys
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from mapping import mapping
# np.set_printoptions(threshold=sys.maxsize)

def drawSame():
    front, left, top = np.zeros((200,200)), np.zeros((200,200)), np.zeros((200,200))
    left[50:150, 50:150] = (left + 1)[50:150, 50:150]
    top[50:150, 25:175] = (top + 1)[50:150, 25:175]
    front[50:150, 25:175] = (front + 1)[50:150, 25:175]

    return front, left, top


front, left, top = drawSame()

data = mapping(front, left, top, 100, 100, 100, 100)
x, y, z = zip(*data)
# z = list(map(float, z))
# grid_x, grid_y = np.mgrid[min(x):max(x):10j, min(y):max(y):10j]
# grid_z = griddata((x, y), z, (grid_x, grid_y))

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(x, y, z)
# plt.imshow(front)
plt.show()
# plt.imshow(left)
# plt.show()
