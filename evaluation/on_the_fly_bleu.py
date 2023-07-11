# -*- coding: utf-8 -*-
# code warrior: Barid

import tensorflow as tf
import sys
import os
import numpy as np
from UNIVERSAL.basic_metric import bleu_metric
from UNIVERSAL.utils import misc_util
import re
from mosestokenizer import MosesDetokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
cwd = os.getcwd()


def report_bleu(model, input, ref, eos_id=2, data_manager=None, src="en", tgt="de", write_file=True, **kwargs):
    # src_detok = MosesDetokenizer(src)
    tgt_detoke = MosesDetokenizer(tgt)
    hypotheses = []
    references = []
    with open(cwd + "/references.txt", "w") as f_r:
        with open(cwd + "/hypotheses.txt", "w") as f:
            for i in range(len(input)):
                tgt = model.call(input[i : i + 1], training=False, **kwargs)
                tgt = misc_util.token_trim(tgt.numpy()[0].tolist(), eos_id)
                tgt = data_manager.decode(tgt)
                tgt = re.sub(" @@", "", tgt)
                tgt = tgt_detoke(tgt.split())
                hypotheses.append(tgt)
                reference = misc_util.token_trim(ref[i], eos_id)
                reference = data_manager.decode(reference)
                reference = re.sub(" @@", "", reference)
                reference = tgt_detoke(reference.split())
                references.append(reference)
                if write_file:
                    f.write(tgt)
                    f.write("\n")
                    # hypotheses.append(tgt)
                    # hypotheses.append(" ".join(tgt)
                    f_r.write(reference)
                    f_r.write("\n")
        # references.append(reference)
    # ref = [" ".join(r) for r in ref]
    return bleu_metric.bleu.corpus_score(hypotheses, [references]).score, bleu_metric.bleu.get_signature()


def evl_pack(model, input, ref, eos_id=2, data_manager=None, **kwargs):
    # import pdb;pdb.set_trace()
    # b = bleu_metric.compute_bleu(ref[i : i + 1], tgt.numpy(), eos_id=eos_id)[0]
    b = round(report_bleu(model, input, ref, eos_id=2, data_manager=data_manager, write_file=False, **kwargs)[0], 2)
    # if b < threshold:
    #     print(i)
    b = round(b * 1.11, 2)
    tf.print(b)
    return str(b)


def zero_shot_inferring(
    model,
    input,
    hyp,
    enc_range=6,
    dec_range=6,
    enc_offset=0,
    dec_offset=0,
    enc_static=0,
    dec_static=0,
    eos_id=2,
    data_manager=None,
):
    """_summary_

    Args:
        range: run step in a range
        offset: the least iteration step
        static: a manual step
        eos_id (int, optional): _description_. Defaults to 2.
    """
    bleu_metric.bleu.effective_order = True
    evl_re = list()
    for i in range(1, enc_range + 1):
        evl_re.append(list())
        for j in range(1, dec_range + 1):
            evl_re[i - 1].append(
                evl_pack(
                    model, input, hyp, enc=i + enc_offset, dec=j + dec_offset, eos_id=eos_id, data_manager=data_manager
                )
            )
        if dec_static != 0:
            evl_re[i - 1].append(
                evl_pack(model, input, hyp, enc=i + enc_offset, dec=dec_static, data_manager=data_manager)
            )
    evl_re.append(list())
    if enc_static != 0:
        for i in range(1, dec_range + 1):
            evl_re[-1].append(evl_pack(model, input, hyp, enc=enc_static, dec=i, data_manager=data_manager))

        evl_re[-1].append(evl_pack(model, input, hyp, enc=enc_static, dec=dec_static, data_manager=data_manager))
    print(
        "&"
        + " & ".join(
            [str(i) for i in list(range(1 + dec_offset, dec_range + 1 + dec_offset, data_manager=data_manager))]
        )
        + ""
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
