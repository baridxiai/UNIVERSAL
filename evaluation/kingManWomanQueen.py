# -*- coding: utf-8 -*-
# code warrior: Barid

import tensorflow as tf

# import sys
# import os
# import numpy as np


def kingManWomanQueen(model, encode, lang1, lang2):
    king = (model(encode("King")) + lang2, "King")
    queen = (model(encode("Queen")) + lang2, "Queen")
    man = (model(encode("Man")) + lang2, "Man")
    woman = (model(encode("Woman")) + lang2, "Woman")
    # German
    Konig = (model(encode("König")) + lang1, "König")
    Konigin = (model(encode("Königin")) + lang1, "Königin")
    mann = (model(encode("Mann")) + lang1, "Mann")
    frau = (model(encode("Frau")) + lang1, "Frau")

    King = [king, Konig]
    Man = [man, mann]
    Woman = [woman, frau]
    Queen = [queen, Konigin]
    print("| Item| Queen| Königin|")
    for _, k in enumerate(King):
        for _, m in enumerate(Man):
            for _, w in enumerate(Woman):
                # for _,q in enumerate(Queen):
                queen_cos = -tf.keras.losses.cosine_similarity(k[0] - m[0] + w[0], queen[0])
                konigin_cos = -tf.keras.losses.cosine_similarity(k[0] - m[0] + w[0], Konigin[0])
                print(
                    "|"
                    + k[1]
                    + "-"
                    + m[1]
                    + "+"
                    + w[1]
                    + "|"
                    + str(round(queen_cos.numpy()[0], 2))
                    + "|"
                    + str(round(konigin_cos.numpy()[0], 2))
                    + "|"
                )


# kingManWomanQueen(model.embedding_softmax_layer,data_manager.encode)
# kingManWomanQueen(
#     model.embedding_softmax_layer,
#     data_manager.encode,
#     model.lang_encoding([1]),
#     model.lang_encoding([2]),
# )
