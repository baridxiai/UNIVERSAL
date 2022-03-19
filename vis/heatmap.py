# -*- coding: utf-8 -*-
# code warrior: Barid

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import tensorflow as tf
############################################################
vegetables = [
    "The", "[CLPM]_1", "fund", "[CLPM]_3", "owned", "[CLPM]_5", "building",
    "[CLPM]_7", "to", "make", "a", "choice", "."
]
farmers = ["investment", "that", "the", "had"]

# fig, ax = plt.subplots(figsize=(10, 1.5))
fig, ax = plt.subplots()
df = np.array([[
    0.1, 0.7, 0.1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01
], [0.05, 0.15, 0.2, 0.3, 0.15, 0.1, 0.15, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
               [
                   0.01, 0.01, 0.01, 0.05, 0.1, 0.7, 0.2, 0.01, 0.01, 0.01,
                   0.01, 0.01, 0.01
               ],
               [
                   0.01, 0.01, 0.1, 0.01, 0.01, 0.05, 0.05, 0.1, 0.35, 0.2,
                   0.01, 0.01, 0.01
               ]])

# code = [
#     17601,
#     2020,
#     2385,
#     791,
#     350,
#     782,
#     9309,
#     30,
#     596,
#     2578,
#     534,
#     474,
#     1244,
#     31513,
#     16,
#     448,
#     2953,
#     19758,
#     381,
#     3278,
#     9518,
#     383,
#     3142,
#     369,
#     367,
#     4263,
#     12520,
#     383,
#     19966,
#     16,
#     655,
#     2809,
#     1955,
#     23709,
#     335,
#     1014,
#     782,
#     1091,
#     916,
#     383,
#     1345,
#     415,
#     18,
#     2,
# ]
#src
code = [
41, 18511, 22158,  2594,  8763,  2606,  2594,  2891,  3291,
       31737,  2625,  2586, 13701,  4541,  7915,  6960,    14,  2605,
          41,  3029,  3163,  5438,  3444,  2610,  4532,  2795,    62,
        8445,  3102,  3167,  2581,  2594,  4498,  2672,  2795, 16308,
          62, 10029,  3789,  2758,  5272,    16,     2,

]
# code = [
# 4041,  3548,  1284,    45,   644,   804,   360,  1130,   342,
#           401,   380,   689,   470,  4891,  4114,   477, 15199,  2617,
#          8430,   335,   346,  2620,  1431,   363, 15199,   478, 11412,
#           360,   346,  1103,   401,   657,   342,  1561,  1557,   764,
#           360,   359,  2143,   338,   346,  1405,   362,  2367,  2620,
#           335,   472,    18,     2
# ]
# with open("../../lazy_transformer/cka_similarity.json") as json_file:
#ref
ref = [
2960, 28951,  2618,  2627, 14742,    14,  2717,  4541,    16,
        7998, 16135,    66, 23480,  2702,  3295,  4392,  2692,  3688,
        7721,    14, 11059,  3543, 12210,  4977, 19740,  3050,  5803,
        8805,  2614,  6007,    14,  3137,  2751, 10034,  7649,  6725,
          16,     2, 2
        ]
code  = [2709, 3190, 2588, 20637, 2707, 30714, 16047, 80, 14, 2823, 2677, 14727, 6244, 2672, 14073, 2837, 7837, 3525, 2594, 21031, 14, 3606, 3429, 2594, 7329, 2588, 4843, 2581, 2594, 12306, 16, 2,2]
ref = [2784, 11254, 2647, 2693, 23573, 2785, 16047, 80, 8057, 14, 2618, 2679, 28020, 29481, 14, 2772, 2618, 44, 4370, 2682, 2620, 29222, 3042, 2794, 6361, 2773, 2855, 9794, 3670, 2598, 14381, 4130, 2639, 8559, 2831, 16, 2,2,2]
# code = ref
step = 6
vegetables = []
df = []
farmers = []
def sentence_vis(path):
    with open(path) as json_file:
        data = json.load(json_file)
        farmers = list(range(step,0,-1))
    sns.set(rc={'figure.figsize': (16, 16)})
    sns.heatmap(np.reshape(data, (step, 1)),
                cmap="YlGnBu",
                xticklabels="s",
                yticklabels=farmers,
    ax=ax,annot=True,square=True, annot_kws={"fontsize":6},cbar=False)
    return
def token_vis(path):
    with open(path) as json_file:
        data = json.load(json_file)
        vegetables = list(range(len(code)))

        # df.append(data[str(step+1 - i)])
        # vegetables = 1

        farmers = list(range(step,0,-1))
    # import pdb;pdb.set_trace()
    sns.set(rc={'figure.figsize': (16, 16)})

    # for token level ############################################
    sns.heatmap(np.transpose(data, (1, 0)),
    cmap="YlGnBu",
    xticklabels=vegetables,
    yticklabels=farmers,
    ax=ax,annot=True,square=True, annot_kws={"fontsize":4},cbar=False)
    return
# token_vis("../universalTransformer/enc_cka_similarity.json")
# sentence_vis("../universalTransformer/dec_cka_similarity_sentence.json")
#########
# token_vis("/Users/barid/Desktop/lt/lt_6_SOTA/dec_halting_pro.json")
##############
# token_vis("../lazy_transformer/dec_halting_pro.json"
# token_vis("/Users/barid/Desktop/lt/ut_6/enc_cka_similarity.json")
# sentence_vis("../lazy_transformer/enc_cka_similarity_sentence.json")
token_vis("../lazy_transformer/enc_cka_similarity.json")

plt.setp(ax.get_yticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
plt.xlabel("position")
plt.ylabel("step")
fig.tight_layout()
plt.rcParams['savefig.dpi'] = 300
plt.savefig("/Users/barid/Desktop/acl_UT6NO_enc_cka_similarity.png")
plt.show()
#######################################################