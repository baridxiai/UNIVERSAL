# -*- coding: utf-8 -*-
# code warrior: Barid

from mosestokenizer import MosesTokenizer

en_tokenize = MosesTokenizer("en")
de_tokenize = MosesTokenizer("de")
data = [
    "/home/vivalavida/massive_data/data/europarl-v7/europarl-v7.de-en.en",
    "/home/vivalavida/massive_data/data/training-parallel-commoncrawl/commoncrawl.de-en.en",
]
data_1 = open("/home/vivalavida/massive_data/data/europarl-v7/europarl-v7.de-en.en", "r")
data_1 = data_1.read().split("\n")
data_2 = open("/home/vivalavida/massive_data/data/training-parallel-commoncrawl/commoncrawl.de-en.en", "r")
data_2 = data_2.read().split("\n")
data_3 = open("/home/vivalavida/massive_data/data/News_Commentary/news-commentary-v9.de-en.en", "r")
data_3 = data_3.read().split("\n")
with open("/home/vivalavida/massive_data/data/parallel_corpora/europarl-v7+commoncrawl+news.de-en.en", "w") as output:
    t = 0
    for k, v in enumerate(data_3):
        # for index, line in enumerate(input.readlines()):
        str = data_3[k]
        output.write(str)
        output.write("\n")
        str = data_2[k]
        output.write(str)
        output.write("\n")
        str = data_1[k]
        output.write(str)
        output.write("\n")
        t = k
    for k in range(t, len(data_1)):
        str = data_2[k]
        output.write(str)
        output.write("\n")
        str = data_1[k]
        output.write(str)
        output.write("\n")
        t = k
    for k in range(t, len(data_2)):
        str = data_2[k]
        output.write(str)
        output.write("\n")
        t = k

# data = [
#     "/home/vivalavida/massive_data/data/europarl-v7/europarl-v7.de-en.de",
#     "/home/vivalavida/massive_data/data/training-parallel-commoncrawl/commoncrawl.de-en.de"
# ]
data_1 = open("/home/vivalavida/massive_data/data/europarl-v7/europarl-v7.de-en.de", "r")
data_1 = data_1.read().split("\n")
data_2 = open("/home/vivalavida/massive_data/data/training-parallel-commoncrawl/commoncrawl.de-en.de", "r")
data_2 = data_2.read().split("\n")
data_3 = open("/home/vivalavida/massive_data/data/News_Commentary/news-commentary-v9.de-en.de", "r")
data_3 = data_3.read().split("\n")

with open("/home/vivalavida/massive_data/data/parallel_corpora/europarl-v7+commoncrawl+news.de-en.de", "w") as output:
    t = 0
    for k, v in enumerate(data_3):
        # for index, line in enumerate(input.readlines()):
        str = data_3[k]
        output.write(str)
        output.write("\n")
        str = data_2[k]
        output.write(str)
        output.write("\n")
        str = data_1[k]
        output.write(str)
        output.write("\n")
        t = k
    for k in range(t, len(data_1)):
        str = data_2[k]
        output.write(str)
        output.write("\n")
        str = data_1[k]
        output.write(str)
        output.write("\n")
        t = k
    for k in range(t, len(data_2)):
        str = data_2[k]
        output.write(str)
        output.write("\n")
        t = k
