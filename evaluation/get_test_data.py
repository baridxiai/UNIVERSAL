# -*- coding: utf-8 -*-
# code warrior: Barid

# import tensorflow as tf
# import sys
# import os
# import numpy as np

from mosestokenizer import MosesTokenizer


def get_input_and_hyp(input_path, ref_path, add_eos=True, span=30, check_period=True):
    def _read_file(path):
        re = []
        with open(path) as f:
            for k, v in enumerate(f.readlines()):
                if k % span == 0:
                    v = v.strip()
                    # if v[-1] != "." and check_period:
                    #     v += " ."
                    re.append(v + " [EOS]" if add_eos else v)
        return re  # print(v)

    return _read_file(input_path), _read_file(ref_path)


def get_SemEval(input_path, spliter="\t", hyps="de", refs="en"):
    hyps_tok = MosesTokenizer(hyps)
    refs_tok = MosesTokenizer(refs)

    def _read_file(path):
        src = []
        tgt = []
        with open(path) as f:
            for _, v in enumerate(f.readlines()):
                v_src, v_tgt = v.strip().split(spliter)
                src.append(v_src.strip())
                tgt.append(v_tgt.strip())
        return src, tgt  # print(v)

    return _read_file(input_path)
