# -*- coding: utf-8 -*-
# code warrior: Barid
##########
import re, seaborn as sns
import numpy as np
from numpy import load
import pandas as pd

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
tsne_result = load("/Users/barid/Documents/workspace/alpha/cross-lingual-masking/tsne.data.npy")
# tsne_result_df = pd.DataFrame({'pca_1': tsne_result[:,0], 'pca_2': tsne_result[:,1], 'label':tsne_result[:,2]})

x = tsne_result[:,0]
y = tsne_result[:,1]
z = tsne_result[:,2]
# c = tsne_result[:,3]
d = np.where(tsne_result[:,3] == 1.,'o','+').tolist()
# axes instance
# fig = plt.figure(figsize=(6,6))
# ax = Axes3D(fig, auto_add_to_figure=False)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# get colormap from seaborn
cmap = ListedColormap(sns.color_palette("husl", 256).as_hex())

# plot
for i in range(len(tsne_result)):
    ax.scatter(tsne_result[i,0], tsne_result[i,1], tsne_result[i,2], s=5, marker='o' if tsne_result[i,3] == 0. else '^', color='#db77a3' if tsne_result[i,3] == 0. else '#86cef5', alpha=0.9)
ax.set_xlabel('tsne_x')
ax.set_ylabel('tsne_y')
ax.set_zlabel('tsne_z')

# legend
# plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)

plt.show()
# save
# plt.savefig("scatter_hue", bbox_inches='tight')