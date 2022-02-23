# -*- coding: utf-8 -*-
# code warrior: Barid
import tensorflow as tf


class Sum_MetricLayer(tf.keras.layers.Layer):
    def __init__(self, name="custom_mean"):
        self.sum_fn = tf.keras.metrics.Sum()
        self.metric_name = name
        super(Sum_MetricLayer, self).__init__(name=name)

    def build(self, input_shape):
        """"Builds metric layer."""
        super(Sum_MetricLayer, self).build(input_shape)

    def call(
        self, inputs, penalty=1,
    ):
        # self.add_loss(inputs * penalty)
        self.add_metric(self.sum_fn(inputs), name=self.metric_name, aggregation="mean")
        return inputs
