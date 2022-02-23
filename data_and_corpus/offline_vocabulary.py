# -*- coding: utf-8 -*-
# code warrior: Barid

def read_offine_vocabulary(path):
    vocabulary = []
    with open(path, "r") as f:
        for k, v in enumerate(f.readlines()):
            vocabulary.append(int(v.strip()))
    return vocabulary
