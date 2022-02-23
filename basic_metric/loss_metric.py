# -*- coding: utf-8 -*-
# code warrior: Barid
import tensorflow as tf
from UNIVERSAL.utils import padding_util


def bilogits_loss_function(true, pred, mask_id=0, smoothing=0.1, pre_sum=False, weight_true=True):
    """Short summary.
    Args:
        pred (type): Description of parameter `pred`.
        true (type): Description of parameter `true`.

    Returns:
        type: Description of returned object.

    """

    logits, labels = pred, true
    with tf.name_scope("smoothing_cross_entropy"):
        # xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
        #                                                    labels=labels)
        xentropy = tf.keras.losses.binary_crossentropy(labels, logits)
    if weight_true:
        weights = tf.cast(tf.not_equal(tf.reduce_sum(true, -1), 0), tf.float32)
    else:
        weights = tf.cast(tf.not_equal(tf.reduce_sum(pred, -1), 0), tf.float32)
    xentropy *= weights
    if pre_sum:
        loss = tf.reduce_sum(input_tensor=xentropy) / tf.reduce_sum(input_tensor=weights)
    else:
        loss = tf.reduce_sum(input_tensor=xentropy, axis=-1) / tf.reduce_sum(input_tensor=weights, axis=-1)
        # loss = tf.expand_dims(loss, -1)
    return loss


def logits_loss_function(true, pred, mask_id=0, smoothing=0.1, pre_sum=False):
    """Short summary.
    Args:
        pred (type): Description of parameter `pred`.
        true (type): Description of parameter `true`.

    Returns:
        type: Description of returned object.

    """

    logits, labels = pred, true
    with tf.name_scope("smoothing_cross_entropy"):
        xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    weights = tf.cast(tf.not_equal(tf.reduce_sum(labels, -1), 0), tf.float32)
    xentropy *= weights
    if pre_sum:
        loss = tf.reduce_sum(input_tensor=xentropy) / tf.reduce_sum(input_tensor=weights)
    else:
        loss = tf.reduce_sum(input_tensor=xentropy, axis=-1) / tf.reduce_sum(input_tensor=weights, axis=-1)
        # loss = tf.expand_dims(loss, -1)
    return loss


def spare_crossentropy(true, pred, mask_id=0, vocab_size=25000):
    if len(true.get_shape().as_list()) > 1:
        logits, labels = padding_util.pad_tensors_to_same_length(pred, true)
    else:
        logits, labels = pred, true
    mask = tf.math.logical_not(tf.math.equal(labels, 0))
    loss_ = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


def onehot_loss_function(true, pred, mask_id=0, smoothing=0.1, vocab_size=25000, pre_sum=False):
    """Short summary.
    Args:
        pred (type): Description of parameter `pred`.
        true (type): Description of parameter `true`.

    Returns:
        type: Description of returned object.

    """
    if len(true.get_shape().as_list()) > 1:
        logits, labels = padding_util.pad_tensors_to_same_length(pred, true)
    else:
        logits, labels = pred, true
    with tf.name_scope("ns_smoothing_cross_entropy"):
        if len(logits.get_shape().as_list()) <= 2:
            logits = tf.one_hot(tf.cast(logits, tf.int32), depth=vocab_size)
        confidence = 1.0 - smoothing
        low_confidence = (1.0 - confidence) / tf.cast(vocab_size - 1, tf.float32)
        soft_targets = tf.one_hot(
            tf.cast(labels, tf.int32), depth=vocab_size, on_value=confidence, off_value=low_confidence,
        )
        xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=soft_targets,)
        normalizing_constant = -(
            confidence * tf.math.log(confidence)
            + tf.cast(vocab_size - 1, tf.float32) * low_confidence * tf.math.log(low_confidence + 1e-20)
        )
        xentropy -= tf.cast(normalizing_constant, xentropy.dtype)
        # xentropy = tf.keras.losses.categorical_crossentropy(soft_targets,logits, from_logits=True)
        weights = tf.cast(tf.not_equal(labels, 0), xentropy.dtype)
    # weights = tf.cast(tf.not_equal(labels, 0), tf.float32)
    xentropy *= weights
    if pre_sum:
        loss = tf.reduce_sum(input_tensor=xentropy) / tf.reduce_sum(input_tensor=weights)
    else:
        loss = tf.reduce_sum(input_tensor=xentropy, axis=-1) / tf.reduce_sum(input_tensor=weights, axis=-1)
    return loss


def sampled_loss_function(
    true, pred, weights, bias=None, mask_id=0, smoothing=0.1, vocab_size=25000, pre_sum=False,
):
    """Short summary.
    Args:
        pred (type): Description of parameter `pred`.
        true (type): Description of parameter `true`.

    Returns:
        type: Description of returned object.

    """
    if bias is None:
        bias = tf.zeros([vocab_size])
    logits, labels = padding_util.pad_tensors_to_same_length(pred, true)
    # Calculate smoothing cross entropy
    if len(logits.get_shape().as_list()) <= 2:
        logits = tf.one_hot(tf.cast(logits, tf.int32), depth=vocab_size)
    confidence = 1.0 - smoothing
    low_confidence = (1.0 - confidence) / tf.cast(vocab_size - 1, tf.float32)
    dim = tf.shape(logits)[2]
    xentropy = tf.nn.sampled_softmax_loss(
        weights, bias, tf.reshape(labels, [-1, 1]), tf.reshape(logits, [-1, dim]), 1000, vocab_size,
    )
    normalizing_constant = -(
        confidence * tf.math.log(confidence)
        + tf.cast(vocab_size - 1, tf.float32) * low_confidence * tf.math.log(low_confidence + 1e-20)
    )
    xentropy -= tf.cast(normalizing_constant, xentropy.dtype)
    # xentropy = tf.keras.losses.categorical_crossentropy(soft_targets,logits, from_logits=True)
    weights = tf.cast(tf.not_equal(labels, 0), xentropy.dtype)
    xentropy *= weights
    if pre_sum:
        loss = tf.reduce_sum(input_tensor=xentropy) / tf.reduce_sum(input_tensor=weights)
    else:
        loss = tf.reduce_sum(input_tensor=xentropy, axis=-1) / tf.reduce_sum(input_tensor=weights, axis=-1)
        # loss = tf.expand_dims(loss, -1)
    return loss


class SimLoss_layer(tf.keras.layers.Layer):
    def __init__(self, penalty=1, name="sim_loss", log_loss=True, **kwargs):
        super(SimLoss_layer, self).__init__(**kwargs)
        self.penalty = penalty
        self.log_loss = log_loss
        if self.log_loss:
            self.mean_fn = tf.keras.metrics.Mean(name)

    def build(self, input_shape):
        super(SimLoss_layer, self).build(input_shape)

    def get_config(self):
        return {
            "vocab_size": self.vocab_size,
            "label_smoothing": self.label_smoothing,
        }

    def call(self, inputs):
        targets, logits = inputs[0], inputs[1]
        weights = tf.cast(tf.not_equal(tf.reduce_sum(targets, -1), 0), tf.float32)
        loss = 1 + tf.reduce_sum(tf.keras.losses.cosine_similarity(targets, logits) * weights) / tf.reduce_sum(
            input_tensor=weights
        )

        self.add_loss(loss * self.penalty)
        if self.log_loss:
            m = self.mean_fn(loss)
            self.add_metric(m)
        return logits


class Logits_CrossEntropy_layer(tf.keras.layers.Layer):
    def __init__(self, vocab_size, label_smoothing, penalty=1, name="custom", log_loss=True, **kwargs):
        super(Logits_CrossEntropy_layer, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.label_smoothing = label_smoothing
        self.penalty = penalty
        self.log_loss = log_loss
        if self.log_loss:
            self.mean_fn = tf.keras.metrics.Mean(name)

    def build(self, input_shape):
        super(Logits_CrossEntropy_layer, self).build(input_shape)

    def get_config(self):
        return {
            "vocab_size": self.vocab_size,
            "label_smoothing": self.label_smoothing,
        }

    def call(self, inputs):
        targets, logits = inputs[0], inputs[1]
        loss = logits_loss_function(
            targets, logits, smoothing=self.label_smoothing, vocab_size=self.vocab_size, pre_sum=True,
        )
        self.add_loss(loss * self.penalty)
        if self.log_loss:
            m = self.mean_fn(loss)
            self.add_metric(m)
        return logits


class Bi_CrossEntropy_layer(tf.keras.layers.Layer):
    def __init__(self, label_smoothing=0.1, penalty=1, name="bi_cross", log_loss=True, **kwargs):
        super(Bi_CrossEntropy_layer, self).__init__(name=name, **kwargs)
        self.label_smoothing = label_smoothing
        self.penalty = penalty
        self.log_loss = log_loss
        if self.log_loss:
            self.mean_fn = tf.keras.metrics.Mean(name)

    def build(self, input_shape):
        super(Bi_CrossEntropy_layer, self).build(input_shape)

    def get_config(self):
        return {
            "label_smoothing": self.label_smoothing,
        }

    def weighted_loss(self, loss, x):
        weights = tf.squeeze(tf.not_equal(x, 0), -1)
        weights = tf.cast(weights, loss.dtype)
        loss *= weights
        loss = tf.reduce_sum(loss) / tf.reduce_sum(weights)
        return loss

    def call(self, inputs):
        targets, logits = inputs[0], inputs[1]
        targets = tf.reshape(targets, [-1, 1])
        logits = tf.reshape(logits, [-1, 1])
        loss = tf.keras.losses.binary_crossentropy(targets, logits, label_smoothing=self.label_smoothing)
        loss = self.weighted_loss(loss, logits)
        # loss = tf.reduce_mean(loss)
        self.add_loss(loss)
        if self.log_loss:
            m = self.mean_fn(loss)
            self.add_metric(m)
        return logits


class Loss_MetricLayer(tf.keras.layers.Layer):
    """Custom a layer of metrics for logging loss."""

    def __init__(self, name="custom_loss"):
        # self.mean_fn = tf.keras.metrics.Mean(name)
        self.metric_name = name
        super(Loss_MetricLayer, self).__init__(name=name)

    def build(self, input_shape):
        """"Builds metric layer."""
        super(Loss_MetricLayer, self).build(input_shape)

    def get_config(self):
        return {"vocab_size": self.vocab_size}

    def call(self, inputs, penalty=1, loss=False):
        if loss:
            self.add_loss(inputs * penalty)
        self.add_metric(inputs, name=self.metric_name, aggregation="mean")
        return inputs
