from edge import return_3_view_img
from image import drawSame
from mapping import mapping
from matplotlib import pyplot as plt

front, left, top = drawSame()
return_3_view_img()
data = mapping(front, left, top, 100, 100, 100, 100)
mid = front.shape[0]/2
x, y, z = zip(*data)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(x, y, z)
plt.show()