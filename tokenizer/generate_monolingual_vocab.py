# -*- coding: utf-8 -*-
# code warrior: Barid

# import tensorflow as tf
with open("./vocabulary/DeEn_60000/statistic_monolingual_En60000.statistic", "r") as f:
    data = []
    for k, v in enumerate(f.readlines()):
        v = v.strip()
        v = v.split("@")
        data.append((v[0],v[1]))

    data = sorted(data,key=lambda x: x[1],reverse=True)
    with open("./vocabulary/DeEn_60000/vocab_monolingual_En60000.vocab", "w") as output:
        # for k, v in enumerate(f.readlines()):
        for k, v in enumerate(data):
            # v = v.strip()
            # v = v.split("@")
            if int(v[1]) > 50000:
            # if int(k) < 5000:
                output.write("%d\n" % int(v[0]))
with open("./vocabulary/DeEn_60000/statistic_monolingual_De60000.statistic", "r") as f:
    data = []
    for k, v in enumerate(f.readlines()):
        v = v.strip()
        v = v.split("@")
        data.append((v[0],v[1]))
    data = sorted(data,key=lambda x: x[1],reverse=True)
    with open("./vocabulary/DeEn_60000/vocab_monolingual_De60000.vocab", "w") as output:
        for k, v in enumerate(data):
            # v = v.strip()
            # v = v.split("@")
            if int(v[1]) > 30000:
            # if int(k) < 5000:
                output.write("%d\n" % int(v[0]))
