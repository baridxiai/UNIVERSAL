# -*- coding: utf-8 -*-
# code warrior: Barid

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import pandas as pd


fig = plt.figure()
ax = fig.gca(projection="3d")


def org_vis(path):
    with open(path) as json_file:
        a = np.array(json.load(json_file)[0])
        df = pd.DataFrame(a, columns=["X", "Y", "Z"])
        # df["X"] = pd.Categorical(df["X"])
        # df["X"] = df["X"].cat.codes

        # Make the plot
        ax.plot_trisurf(df["Y"], df["X"], df["Z"], cmap=plt.cm.viridis, linewidth=0.2)
        # vegetables = list(range(len(code)))

        # df.append(data[str(step+1 - i)])
        # vegetables = 1

        # farmers = list(range(step,0,-1))
    # import pdb;pdb.set_trace()
    # data = np.array(data[0])
    # ax.plot_surface(x,y,z)


org_vis("../lazy_transformer/enc_orgData.json")
fig.tight_layout()
plt.show()
