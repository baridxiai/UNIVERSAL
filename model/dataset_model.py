# -*- coding: utf-8 -*-
# code warrior: Barid
##########
import os
import sys

import numpy as np
import tensorflow as tf
from UNIVERSAL.data_and_corpus import (data_manager, dataset_preprocessing,
                                       offline_corpus)

def preprocessed_dataset(offline,vocab_path,bpe_path,configuration,encode_fn=None,dev_path=None,shuffle=40):
    training_samples = offline_corpus.offline_multi(offline)
    dataManager = data_manager.DatasetManager_multi(
        [
            vocab_path,
            bpe_path,
        ],
        training_samples,
        cut_length=configuration.parameters["max_sequence_length"],
        tokenization=False,
        dev_set=offline_corpus.offline_multi(dev_path if dev_path is not None else offline),
        mono=True,
    )
    dataset = dataManager.get_raw_train_dataset(shuffle)
    preprocessed_dataset = dataset_preprocessing.prepare_training_input_MONO(
        dataset,
        configuration.parameters["batch_size"],
        configuration.parameters["max_sequence_length"],
        min_boundary=8,
        filter_min=1,
        filter_max=configuration.parameters["max_sequence_length"],
        tf_encode=encode_fn,
        shuffle=shuffle,
    )
    return preprocessed_dataset, dataManager

def dev_dataset(dataManager,configuration, encode_fn):
    dataset = dataManager.get_raw_dev_dataset()
    preprocessed_dataset = dataset_preprocessing.prepare_training_input_MONO(
        dataset,
        configuration.parameters["batch_size"],
        configuration.parameters["max_sequence_length"],
        min_boundary=50,
        filter_min=1,
        filter_max=configuration.parameters["max_sequence_length"],
        tf_encode=encode_fn,
    )
    # preprocessed_dataset = dataset.padded_batch(2,padding_values=0)
    # dataset = dataset.map(encode_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # preprocessed_dataset = dataset.padded_batch(2,padding_values=0)
    return preprocessed_dataset