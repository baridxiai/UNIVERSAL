# -*- coding: utf-8 -*-
# code warrior: Barid

from UNIVERSAL.basic_metric import bleu_metric
from UNIVERSAL.vis import linePlotWrapper
import numpy as np
import statistics
import argparse


def get_input_and_hyp(input_path, ref_path):
    def _read_file(path):
        re = []
        with open(path) as f:
            for k, v in enumerate(f.readlines()):
                v = v.strip()
                re.append(v)
        return re  # print(v)

    return _read_file(input_path), _read_file(ref_path)


def report_bucket(hyp_path, ref_path):
    bleu_metric.bleu.effective_order=True
    hyps, refs = get_input_and_hyp(hyp_path, ref_path)
    bucket = [[0] for _ in range(75)]
    # print(bleu_metric.bleu.corpus_score(hypotheses=hyps,references=[refs]))
    for i in range(len(hyps)):
        index = len(refs[i].split())
        b = round(bleu_metric.bleu.sentence_score(hyps[i], [refs[i]]).score,2)
        if  bucket[index][0] ==0:
            bucket[index][0] = b
        else:
            bucket[index].append(b)
    hyps, refs = get_input_and_hyp( "/Users/barid/Documents/workspace/alpha/lazy_transformer/hypotheses.txt", ref_path)
    report = [np.mean(bucket[i]) for i in range(len(bucket))]
    for i in range(len(hyps)):
        index = len(refs[i].split())
        b = round(bleu_metric.bleu.sentence_score(hyps[i], [refs[i]]).score,2)
        if  bucket[index][0] ==0:
            bucket[index][0] = b
        else:
            bucket[index].append(b)
    report_2 = [np.mean(bucket[i]) for i in range(len(bucket))]
    print(np.mean(report[report!=0]))
    linePlotWrapper.showline([report,report_2],x_label='length',y_label="BLEU",title="translation statistics",label=["base-LT","base-UT"])
    # return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref',type=str,required=True)
    parser.add_argument('--hyp',type=str,required=True)
    arg = parser.parse_args()
    refs = arg.ref
    hyps = arg.hyp
    report_bucket(hyps,refs)
