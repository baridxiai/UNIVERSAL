import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility
def showline(list_data, x_label="", y_label="", title=""):
    fig, ax = plt.subplots()
    yy = list_data
    xx = list(range(len(yy)))
    ax.plot(xx, yy)

    ax.set(xlabel=x_label, ylabel=y_label, title=title)
    ax.grid()

    plt.scatter(xx, yy, c="g", alpha=0.5, label="Luck")
    plt.xlabel("Leprechauns")
    plt.ylabel("Gold")
    plt.legend(loc="upper left")
    plt.show()
