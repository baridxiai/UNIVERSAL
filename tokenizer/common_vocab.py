# -*- coding: utf-8 -*-
# code warrior: Barid

def read_vocab(path):
    vocab = []
    with open(path, "r") as f:
        for k, v in enumerate(f.readlines()):
            vocab.append(v.strip())
    return vocab
