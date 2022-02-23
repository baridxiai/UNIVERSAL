# -*- coding: utf-8 -*-
# code warrior: Barid
import tensorflow as tf


class BOW(tf.keras.layers.Layer):
    """
        compute the bag-of-words loss
    """

    def __init__(self, vocab_size, name="BOW"):
        self.log = tf.keras.metrics.Mean(name)
        self.vocab_size = vocab_size
        super(BOW, self).__init__()

    def call(self, pre, label, sentence=True):
        targets_ids, input_logits = label, pre
        targets_ids = tf.one_hot(targets_ids, self.vocab_size)
        if sentence:
            targets_ids = tf.reduce_sum(targets_ids, -2)
            input_ids = tf.reduce_sum(tf.nn.relu(input_logits), -2)
        targets_ids = tf.cast(tf.not_equal(targets_ids, 0), tf.float32)
        input_ids = tf.nn.sigmoid(input_ids)
        loss = tf.keras.losses.binary_crossentropy(
            targets_ids, input_ids, label_smoothing=0.1
        )
        loss = tf.reduce_mean(loss)
        self.add_loss(loss)
        m = self.log(loss)
        self.add_metric(m)
        return loss
