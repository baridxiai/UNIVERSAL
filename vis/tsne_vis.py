# -*- coding: utf-8 -*-
# code warrior: Barid
import seaborn as sns
from numpy import load
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
tsne_result = load("/Users/barid/Documents/workspace/alpha/cross-lingual-masking/tsne.data.npy")
tsne_result_df = pd.DataFrame({'tsne_x': tsne_result[:,0], 'tsne_y': tsne_result[:,1], 'label':tsne_result[:,2]})
# bins = [float(0.1*i) for i in range(0,11)]
# bins_label = [str(0.1*i)[:3] for i in range(1,11)]
tsne_result_df["languages"] = pd.cut(tsne_result_df["label"], bins = 2,labels=["De","En"])
ax = sns.scatterplot(x='tsne_x', y='tsne_y', hue='languages', data=tsne_result_df,palette="deep",
    legend="full",s=15,
    alpha=0.5)
# plt.title("XLM+OURS$_{v2}$")
# plt.title("XLM")
plt.legend(prop={'size': 15})
plt.show()
