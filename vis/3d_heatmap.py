import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colorbar
from matplotlib import cm
import json

viridis = cm.get_cmap('plasma', 8) #Our color map

def cuboid_data(center, size=(1,1,1)):
    # code taken from
    # http://stackoverflow.com/questions/30715083/python-plotting-a-wireframe-3d-cuboid?noredirect=1&lq=1
    # suppose axis direction: x: to left; y: to inside; z: to upper
    # get the (left, outside, bottom) point
    o = [a - b / 2 for a, b in zip(center, size)]
    # get the length, width, and height
    l, w, h = size
    x =  np.array([[o[0], o[0] + l, o[0] + l, o[0], o[0]],      # x coordinate of points in bottom surface
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],                # x coordinate of points in upper surface
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],                # x coordinate of points in outside surface
         [o[0], o[0] + l, o[0] + l, o[0], o[0]]])               # x coordinate of points in inside surface
    y =  np.array([[o[1], o[1], o[1] + w, o[1] + w, o[1]],      # y coordinate of points in bottom surface
         [o[1], o[1], o[1] + w, o[1] + w, o[1]],                # y coordinate of points in upper surface
         [o[1], o[1], o[1], o[1], o[1]],                        # y coordinate of points in outside surface
         [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w]])   # y coordinate of points in inside surface
    z =  np.array([[o[2], o[2], o[2], o[2], o[2]],              # z coordinate of points in bottom surface
         [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],    # z coordinate of points in upper surface
         [o[2], o[2], o[2] + h, o[2] + h, o[2]],                # z coordinate of points in outside surface
         [o[2], o[2], o[2] + h, o[2] + h, o[2]]])               # z coordinate of points in inside surface
    return x, y, z

def plotCubeAt(pos=(0,0,0), c="b", alpha=0.1, ax=None):
    # Plotting N cube elements at position pos
    if ax !=None:
        X, Y, Z = cuboid_data( (pos[0],pos[1],pos[2]) )
        ax.plot_surface(X, Y, Z, color=c, rstride=1, cstride=1, alpha=alpha)

def plotMatrix(ax, x, y, z, data, cmap=viridis, cax=None, alpha=0.1):
    # plot a Matrix
    norm = matplotlib.colors.Normalize(vmin=data.min(), vmax=data.max())
    colors = lambda i,j,k : matplotlib.cm.ScalarMappable(norm=norm,cmap = cmap).to_rgba(data[i,j,k])
    for i, xi in enumerate(x):
            for j, yi in enumerate(y):
                for k, zi, in enumerate(z):
                    if data[i,j,k] != 0.:
                        plotCubeAt(pos=(xi, yi, zi), c=colors(i,j,k), alpha=alpha,  ax=ax)



    if cax !=None:
        cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
        cbar.set_ticks(np.unique(data))
        # set the colorbar transparent as well
        cbar.solids.set(alpha=alpha)

with open("../universalTransformer/enc_orgData.json") as json_file:
    data = json.load(json_file)
data = np.array(data)
x,y,z = data.shape
x = np.array(range(0,x))
y = np.array(range(1,y+1))
z = np.array(range(1,z+1))
# data_value = np.random.randint(1,4, size=(len(x), len(y), len(z)) )
# print(data_value.shape)

fig = plt.figure(figsize=(10,4))
ax = plt.axes(projection='3d')
# ax = fig.add_axes([0.1, 0.1, 0.7, 0.8], projection='3d')

# ax_cb = fig.add_axes([0.8, 0.3, 0.05, 0.45])
# ax.set_aspect('auto')

# import pdb;pdb.set_trace()
plotMatrix(ax, x, y, z, data, cmap=viridis, cax = None)

plt.show()