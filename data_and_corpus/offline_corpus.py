# -*- coding: utf-8 -*-
# code warrior: Barid
import tensorflow as tf


def offline(src_tgt):
    print("offline mode")
    src, tgt = src_tgt
    dataset_0 = tf.data.TextLineDataset(src)
    dataset_1 = tf.data.TextLineDataset(tgt)
    train_examples = tf.data.Dataset.zip((dataset_0, dataset_1))
    return train_examples
