# -*- coding: utf-8 -*-
# code warrior: Barid

import tensorflow as tf
import sys
import os
import numpy as np
from UNIVERSAL.basic_metric import bleu_metric

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def evl_pack(model, input, ref, threshold=5.0, eos_id=2, **kwargs):
    bleu = 0.0
    for i in range(len(input)):
        tgt = model.call(input[i : i + 1], training=False, **kwargs)
        b = bleu_metric.compute_bleu(ref[i : i + 1], tgt.numpy(), eos_id=eos_id)[0]
        if b < threshold:
            print(i)
        print(b)
        bleu += b
    return str(round(bleu / float(len(input)), 2))


def zero_shot_inferring(
    model, input, hyp, enc_range=6, dec_range=6, enc_offset=0, dec_offset=0, enc_static=0, dec_static=0, eos_id=2
):
    evl_re = list()
    for i in range(1, enc_range + 1):
        evl_re.append(list())
        for j in range(1, dec_range + 1):
            evl_re[i - 1].append(evl_pack(model, input, hyp, enc=i + enc_offset, dec=j + dec_offset, eos_id=eos_id))
        if dec_static != 0:
            evl_re[i - 1].append(evl_pack(model, input, hyp, enc=i + enc_offset, dec=dec_static))
    evl_re.append(list())
    if enc_static != 0:
        for i in range(1, dec_range + 1):
            evl_re[-1].append(evl_pack(model, input, hyp, enc=enc_static, dec=i))

        evl_re[-1].append(evl_pack(model, input, hyp, enc=enc_static, dec=dec_static))
    print(
        "&" + " & ".join([str(i) for i in list(range(1 + dec_offset, dec_range + 1 + dec_offset))]) + ""
        if dec_static == 0
        else " &" + str(dec_static)
    )
    if enc_static != 0:
        for i in range(1, len(evl_re)):
            print(str(i + enc_offset) + "&" + " & ".join(evl_re[i - 1]) + "\\" + "\\")
        print(str(enc_static) + "&" + " & ".join(evl_re[-1]) + "\\" + "\\")
    else:
        for i in range(1, len(evl_re) + 1):
            print(str(i + enc_offset) + "&" + " & ".join(evl_re[i - 1]) + "\\" + "\\")
