# -*- coding: utf-8 -*-
# code warrior: Barid
import tensorflow as tf
from UNIVERSAL.utils import padding_util


def padded_accuracy(labels, logits):
    """Percentage of times that predictions matches labels on non-0s."""
    with tf.name_scope("padded_accuracy"):
        logits, labels = padding_util.pad_tensors_to_same_length(logits, labels)
        weights = tf.cast(tf.not_equal(labels, 0), tf.float32)
        if len(logits.get_shape().as_list()) == 3:
            outputs = tf.cast(tf.argmax(logits, axis=-1), tf.int32)
        else:
            outputs = logits
        padded_labels = tf.cast(labels, tf.int32)
        return tf.cast(tf.equal(outputs, padded_labels), tf.float32), weights


def padded_accuracy_topk(labels, logits, k):
    """Percentage of times that top-k predictions matches labels on non-0s."""
    with tf.name_scope("padded_accuracy_topk"):
        logits, labels = padding_util.pad_tensors_to_same_length(logits, labels)
        weights = tf.cast(tf.not_equal(labels, 0), tf.float32)
        effective_k = tf.minimum(k, tf.shape(logits)[-1])
        _, outputs = tf.nn.top_k(logits, k=effective_k)
        outputs = tf.cast(outputs, tf.int32)
        padded_labels = tf.cast(labels, tf.int32)
        padded_labels = tf.expand_dims(padded_labels, axis=-1)
        padded_labels += tf.zeros_like(outputs)  # Pad to same shape.
        same = tf.cast(tf.equal(outputs, padded_labels), tf.float32)
        same_topk = tf.reduce_sum(same, axis=-1)
        return same_topk, weights


def padded_accuracy_top5(labels, logits):
    return padded_accuracy_topk(labels, logits, 5)


def padded_accuracy_top1(labels, logits):
    return padded_accuracy_topk(labels, logits, 1)


def padded_sequence_accuracy(labels, logits):
    """Percentage of times that predictions matches labels everywhere (non-0)."""
    with tf.name_scope("padded_sequence_accuracy"):
        logits, labels = padding_util.pad_tensors_to_same_length(logits, labels)
        weights = tf.cast(tf.not_equal(labels, 0), tf.float32)
        if len(logits.get_shape().as_list()) == 3:
            outputs = tf.cast(tf.argmax(logits, axis=-1), tf.int32)
        else:
            outputs = logits
        padded_labels = tf.cast(labels, tf.int32)
        not_correct = (
            tf.cast(tf.not_equal(outputs, padded_labels), tf.float32) * weights
        )
        axis = list(range(1, len(outputs.get_shape())))
        correct_seq = 1.0 - \
            tf.minimum(1.0, tf.reduce_sum(not_correct, axis=axis))
        return correct_seq, tf.constant(1.0)


class Word_Accuracy_Metric(tf.keras.layers.Layer):
    def __init__(self, name):
        self.trainable = False
        self.mean = tf.keras.metrics.Mean(name)
        super(Word_Accuracy_Metric, self).__init__(name=name)

    def call(self, inputs):
        y_true, y_pred = inputs[0], inputs[1]
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)
        value = padded_accuracy(y_true, y_pred)
        m = self.mean(value)
        self.add_metric(m)
        return y_pred


class Word_Accuracy_Layer(tf.keras.layers.Layer):
    def __init__(self, n):
        super(Word_Accuracy_Layer, self).__init__()
        self.n = n
        self.trainable = False
        self.mean = tf.keras.metrics.Mean(self.n)

    def call(self, inputs):
        targets, logits = inputs[0], inputs[1]
        m = padded_accuracy(targets, logits)
        m = self.mean(m)
        self.add_metric(m)
        return logits


class Sentence_Accuracy_Metric(tf.keras.layers.Layer):
    def __init__(self, name="sentence_acc"):
        self.mean = tf.keras.metrics.Mean(name)
        super(Sentence_Accuracy_Metric, self).__init__(name=name)

    def call(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)
        value, _ = padded_sequence_accuracy(y_true, y_pred)
        value = self.mean(value)
        self.add_metric(value)
        return y_pred


class Word_top5_Accuracy_Metric(tf.keras.layers.Layer):
    def __init__(self, name):
        super(Word_top5_Accuracy_Metric, self).__init__(name=name)
        self.mean = tf.keras.metrics.Mean(name)

    def call(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)
        value, _ = padded_accuracy_top5(y_true, y_pred)
        value = self.mean(value)
        self.add_metric(value)
        return y_pred
