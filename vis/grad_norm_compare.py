# -*- coding: utf-8 -*-
# code warrior: Barid
import numpy as np
import matplotlib.pyplot as plt
import json
import seaborn as sns
import pandas as pd

sns.set_theme(style="whitegrid")
with open("/Users/barid/Desktop/lt/ut_grad_norm.json") as json_file:
    ut = json.load(json_file)
    # for i in range(len(ut)):
    #  ut_y = ut[i][0]
    #  ut_x = ut[i][1]
with open("/Users/barid/Desktop/lt/lt_grad_norm.json") as json_file:
    lt = json.load(json_file)
    # for i in range(len(lt)):
    #  lt_y = lt[i][0]

# Make some fake data.

# Create plots with pre-defined labels.
fig, ax = plt.subplots()

# ax.plot(np.array(ut)[:, 1], np.array(ut)[:, 2], 'k--', label='UT')
# ax.plot(np.array(ut)[:, 1],
#         np.array(lt)[:len(np.array(ut)[:, 2]), 2],
#         'k:',
#         label='LT')

# legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')

# # Put a nicer background color on the legend.
# legend.get_frame().set_facecolor('C0')
data = pd.DataFrame(
    np.transpose([np.array(lt)[:, 2],
     np.array(ut)[:len(np.array(lt)[:, 0]), 2]]),
    np.array(lt)[:, 1],
    columns=["LT", "UT"])
# sns.lineplot(
#     np.array(ut)[:, 1],
#     np.array(ut)[:, 2],
#     x="iteration",
#     y="grad_norm",
# )
# sns.lineplot(
#     np.array(ut)[:, 1],
#     np.array(lt)[:len(np.array(ut)[:, 0]), 2],
#     x="iteration",
#     y="grad_norm",
# )
ax = sns.relplot(data=data, palette="tab10", kind='line',linewidth=2.5)
ax.set(ylabel='grad_norm', xlabel='iteration')
plt.show()