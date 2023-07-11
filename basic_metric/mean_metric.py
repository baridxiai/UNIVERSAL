# -*- coding: utf-8 -*-
# code warrior: Barid
import tensorflow as tf
class Mean_MetricLayer(tf.keras.layers.Layer):

    def __init__(self, name="custom_mean"):
        # self.mean_fn = tf.keras.metrics.Mean(name)
        self.metric_name = name
        super(Mean_MetricLayer, self).__init__(name=name)

    def build(self, input_shape):
        """"Builds metric layer."""
        self.last = tf.Variable(0., trainable=False)
        super(Mean_MetricLayer, self).build(input_shape)


    def call(self, inputs, penalty=1, ):
        # self.add_loss(inputs * penalty)
        re = tf.where(tf.greater(inputs,0.),inputs, self.last)
        self.last.assign(re)
        self.add_metric(re, name=self.metric_name, aggregation="mean")
        return inputs
