# -*- coding: utf-8 -*-
# code warrior: Barid
import tensorflow as tf
from UNIVERSAL.basic_metric import bleu_metric, acc_metric, loss_metric, wer_metric
import functools


class MetricLayer(tf.keras.layers.Layer):
    """Custom a layer of metrics for Transformer model."""

    def __init__(self, trim_id, criteron="all", prefix=""):
        super(MetricLayer, self).__init__()
        self.trim_id = trim_id
        self.metric_mean_fns = []
        self.prefix = prefix
        if prefix != "":
            self.prefix = prefix + "_"

        self.metric_mean_fns = [
            (
                tf.keras.metrics.Mean(self.prefix + "approx_4-gram_bleu"),
                bleu_metric.approx_bleu,
                0,
            ),
            (tf.keras.metrics.Mean(self.prefix + "wer"), wer_metric.wer_score, 1),
            (
                tf.keras.metrics.Mean(self.prefix + "accuracy"),
                acc_metric.padded_accuracy,
                2,
            ),
        ]

    def get_config(self):
        return {}

    def build(self, input_shape):
        """ "Builds metric layer."""
        super(MetricLayer, self).build(input_shape)

    def call(self, inputs):
        targets, logits = inputs[0], inputs[1]
        for mean, fn, index in self.metric_mean_fns:
            if index == 0 or index == 1:
                m = mean(*fn(targets, logits, self.trim_id))
            else:
                m = mean(*fn(targets, logits))
            self.add_metric(m)
        return logits


class CrossEntropy_layer(tf.keras.layers.Layer):
    def __init__(
        self,
        vocab_size,
        label_smoothing,
        mask_id=0,
        penalty=1,
        name="custom",
        log_loss=True,
        type="one_hot",
        **kwargs
    ):
        super(CrossEntropy_layer, self).__init__(name=name)
        self.vocab_size = vocab_size
        self.label_smoothing = label_smoothing
        self.penalty = penalty
        self.log_loss = log_loss
        self.mask_id = mask_id
        if self.log_loss:
            self.mean_fn = tf.keras.metrics.Mean(name)
        self.trainble = False
        self.type = type

    def build(self, input_shape):
        super(CrossEntropy_layer, self).build(input_shape)

    def get_config(self):
        return {
            "vocab_size": self.vocab_size,
            "label_smoothing": self.label_smoothing,
        }

    def call(
        self,
        inputs,
        weights=None,
        bias=None,
        auto_loss=False,
        NS=False,
        pre_sum=True,
        label_weights=1,
        detail_x=False,
    ):
        targets, logits = inputs[0], inputs[1]
        if self.type == "sparse":
            loss = loss_metric.spare_crossentropy(
                targets,
                logits,
                vocab_size=self.vocab_size,
            )
        else:
            if NS and weights is not None:
                loss = loss_metric.sampled_loss_function(
                    targets,
                    logits,
                    smoothing=self.label_smoothing,
                    vocab_size=self.vocab_size,
                    weights=weights,
                    bias=bias,
                    pre_sum=True,
                    mask_id=self.mask_id,
                )
            else:
                loss = loss_metric.onehot_loss_function(
                    targets,
                    logits,
                    smoothing=self.label_smoothing,
                    mask_id=self.mask_id,
                    vocab_size=self.vocab_size,
                    pre_sum=pre_sum,
                    label_weights=label_weights,
                    detail_x=detail_x,
                )
                if detail_x:
                    loss, xentropy = loss
        if self.log_loss:
            m = self.mean_fn(loss)
            self.add_metric(m)

        if auto_loss:
            self.add_loss(loss * self.penalty)
            return logits
        else:
            if detail_x:
                return loss,xentropy
            else:
                return loss


class MeanSquaredError_layer(tf.keras.layers.Layer):
    def __init__(self, log_loss=True, name="custom", **kwargs):
        super(MeanSquaredError_layer, self).__init__(name=name)
        self.layer_fn = tf.keras.losses.mean_squared_error
        self.log_loss = log_loss
        if self.log_loss:
            self.mean_fn = tf.keras.metrics.Mean(name)
        self.trainble = False
        self.type = type

    def build(self, input_shape):
        super(MeanSquaredError_layer, self).build(input_shape)

    def call(self, inputs, auto_loss=False, pre_sum=True, penalty=1):
        weights = 1
        if len(inputs) == 3:
            targets, logits, weights = inputs[0], inputs[1], inputs[2]
        else:
            targets, logits = inputs[0], inputs[1]
        loss = self.layer_fn(targets, logits) * weights
        weights = tf.cast(tf.not_equal(tf.squeeze(targets, -1), 0), loss.dtype)
        loss *= weights
        loss *= penalty
        if pre_sum:
            loss = tf.reduce_sum(input_tensor=loss) / tf.reduce_sum(input_tensor=weights)
        else:
            loss = tf.reduce_sum(input_tensor=loss, axis=-1) / tf.reduce_sum(
                input_tensor=weights, axis=-1
            )
        if self.log_loss:
            m = self.mean_fn(loss)
            self.add_metric(m)

        if auto_loss:
            self.add_loss(loss * self.penalty)
            return loss
        else:
            return loss
