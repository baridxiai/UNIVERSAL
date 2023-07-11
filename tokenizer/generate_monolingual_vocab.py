# -*- coding: utf-8 -*-
# code warrior: Barid

# import tensorflow as tf
import os

c_path = os.getcwd()

de_key = []
with open(c_path + "/vocabulary/DeEn_6k_wiki_balanced/statistic_monolingual_En60000.statistic", "r") as f:
    de_data = []
    for k, v in enumerate(f.readlines()):
        v = v.strip()
        v = v.split(" ")
        de_data.append((v[0], v[1]))
    # data = sorted(de_data, key=lambda x: x[1], reverse=True)
    with open(c_path + "/vocabulary/DeEn_6k_wiki_balanced/vocab_monolingual_En60000.vocab", "w") as output:
        # for k, v in enumerate(f.readlines()):
        for i in range(len(de_data)):
            # v = v.strip()
            # v = v.split("@")
            v = de_data[i]
            if int(v[1]) > 1000:
                # if int(k) < 5000:
                de_key.append(v[0])
                output.write("%d\n" % int(v[0]))
        output.write("%d\n" % int(2))
en_key = []
with open(c_path + "/vocabulary/DeEn_6k_wiki_balanced/statistic_monolingual_De60000.statistic", "r") as f:
    en_data = []
    for k, v in enumerate(f.readlines()):
        v = v.strip()
        v = v.split(" ")
        en_data.append((v[0], v[1]))
    # data = sorted(en_data, key=lambda x: x[1], reverse=True)
    with open(c_path + "/vocabulary/DeEn_6k_wiki_balanced/vocab_monolingual_De60000.vocab", "w") as output:
        for i in range(len(en_data)):
            # v = v.strip()
            # v = v.split("@")
            v = en_data[i]
            if int(v[1]) > 1000:
                # if int(k) < 5000:
                en_key.append(v[0])
                output.write("%d\n" % int(v[0]))
        output.write("%d\n" % int(2))

with open(c_path + "/vocabulary/DeEn_6k_wiki_balanced/vocab_shared.vocab", "w") as output:
    data = set(en_key).intersection(set(de_key))
    for k, v in enumerate(data):
        # v = v.strip()
        # v = v.split("@")
        # if int(v[1]) > 50:
        # if int(k) < 5000:
        output.write("%d\n" % int(v))
