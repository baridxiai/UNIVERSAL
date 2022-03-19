# -*- coding: utf-8 -*-
# code warrior: Barid
import matplotlib.pyplot as plt
import numpy as np


def showline(list_data, x_label="", y_label="", title=""):
    fig, ax = plt.subplots()
    list_data = np.array(list_data)
    yy = list_data[list_data != 0]
    xx = np.array(list(range(len(list_data))))[list_data != 0]
    ax.plot(xx, yy)

    ax.set(xlabel=x_label, ylabel=y_label, title=title)
    ax.grid()

    # fig.savefig("test.png")
    plt.show()
