# -*- coding: utf-8 -*-
# code warrior: Barid
##########
import os
import sys

import numpy as np
import tensorflow as tf
from UNIVERSAL.MLM import preprocess_MLM


def MLM_data_model(parameters):
    # x_input_span, x_output_span, x_span, x_label, lang_ids = preprocess_MLM(inputs, parameters)
    # x_input_span.set_shape([None])
    # x_output_span.set_shape([None])
    # x_span.set_shape([None])
    # x_label.set_shape([None])
    # lang_ids.set_shape([None])

    return (lambda inputs: preprocess_MLM.preprocess_MLM(inputs, parameters))

def Classification_data_model_idLang_label(x,y):
    x = tf.strings.split(x, "@@")
    x = tf.cast(tf.strings.to_number(tf.strings.split(x)), tf.int32)
    x = x.to_tensor()
    ids = tf.cast(x[0, 0:1], tf.int32)
    x = tf.cast(x[1, :256], tf.int32)
    y = tf.one_hot(tf.strings.to_number(y, tf.int32), depth=3, dtype=tf.int32)
    return ((x, y, ids),)

def temperature_sampling(path,alpha=0.5):
    corpora_info, total_alpha = read_corpora_info_langIdPathCount(path,alpha)
    lang_dict=dict()
    sampling = []
    for lang in corpora_info:
        lang_dict[lang[0]] = int(lang[1])
        sampling.append([lang[2],int(lang[3])**alpha/total_alpha])

    return (lang_dict,sampling)

def read_corpora_info_langIdPathCount(path,alpha):
    corpora_info = []
    total = 0
    with open(path) as file:
        for _,line in enumerate(file.readlines()):
            lang = line.strip().split("@@")
            total += int(lang[-1]) ** alpha
            corpora_info.append(lang)
        file.close()

    return corpora_info, total