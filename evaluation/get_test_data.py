# -*- coding: utf-8 -*-
# code warrior: Barid

# import tensorflow as tf
# import sys
# import os
# import numpy as np


def get_input_and_hyp(input_path, ref_path,add_eos=True,span=30,check_period=True):
    def _read_file(path):
        re = []
        with open(path) as f:
            for k, v in enumerate(f.readlines()):
                if k % span == 0:
                    v = v.strip()
                    # if v[-1] != "." and check_period:
                    #     v += " ."
                    re.append(v + " [EOS]" if add_eos else v)
        return re                # print(v)
    return _read_file(input_path), _read_file(ref_path)
