# -*- coding: utf-8 -*-
# code warrior: Barid
#####################
# Requirement:
#   jiwer
######################
from UNIVERSAL.utils import misc_util
import tensorflow as tf
from jiwer import wer


def compute_wer(reference_corpus, translation_corpus, trim_id=0, print_matrix=False):
    try:
        # eos_id = 1
        reference = reference_corpus.numpy().tolist()
        translation = translation_corpus.numpy().tolist()
    except Exception:
        # eos_id = 1
        reference = reference_corpus
        translation = translation_corpus
    score = 0
    num = 0
    for (ref, hyp) in zip(reference, translation):
        if trim_id > 5000:
            hyp = misc_util.token_trim(hyp, trim_id, remider=1)
            ref = misc_util.token_trim(ref, trim_id, remider=1)
        s = wer(" ".join(str(r) for r in ref), " ".join(str(h) for h in hyp))
        score += s
        num += 1
    if num == 0:
        return 0
    return score / num


def wer_score(labels, logits, trim_id=0):
    if len(logits.get_shape().as_list()) > 2:
        logits = tf.argmax(logits, axis=-1)
    else:
        logits = tf.cast(logits, tf.int64)
    labels = tf.cast(labels, tf.int64)
    wer = tf.py_function(compute_wer, [labels, logits, trim_id], tf.float32)
    return wer, tf.constant(1.0)


def wer_fn(labels, logits):
    return wer_score(labels, logits)[0]


class Wer_Metric(tf.keras.metrics.Mean):
    def __init__(self, name):
        self.mean = tf.keras.metrics.Mean(name)
        super(Wer_Metric, self).__init__(name=name)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)
        value, _ = wer_score(y_true, y_pred)
        value = self.mean(value)
        self.add_metric(value)
        return y_pred
