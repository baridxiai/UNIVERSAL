# -*- coding: utf-8 -*-
# code warrior: Barid
import tensorflow as tf

class NaiveNorm(tf.keras.layers.Layer):
    """A naive norm layer without scalling and shiffting.

        ln(x) = (x - μ) / (σ**2 + ϵ)**0.5
    """
    def __init__(
        self, epsilon=1e-9, name="NavieNorm",
    ):
        super(NaiveNorm, self).__init__(name=name)
        self.epsilon = epsilon
    def call(self, inputs):
        o_mean, o_var = tf.nn.moments(inputs, [-1], keepdims=True)
        o = (inputs - o_mean) * tf.math.rsqrt(o_var + self.epsilon)
        return o
class LayerNorm(tf.keras.layers.Layer):
    """
        Layer normalization for transformer, we do that:
            ln(x) = α * (x - μ) / (σ**2 + ϵ)**0.5 + β
    """

    def __init__(
        self, epsilon=1e-9, gamma_initializer="ones", beta_initializer="zeros", mode="linear", name="norm",
    ):
        super(LayerNorm, self).__init__(name=name)
        self.epsilon = epsilon
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        self.mode = mode

    def build(self, input_shape):
        input_dim = input_shape[-1]

        self.gamma_kernel = self.add_weight(shape=(input_dim), name="gamma", initializer=self.gamma_initializer)
        self.beta_kernel = self.add_weight(shape=(input_dim), name="beta", initializer=self.beta_initializer)
        super(LayerNorm, self).build(input_shape)

    def call(self, inputs):
        if self.mode == "tanh":
            filter = tf.cast(tf.not_equal(inputs, 0.0), tf.float32)
            mean, variance = tf.nn.moments(inputs, [-1], keepdims=True)
            tanh_estimator = tf.cast(0.01 * ((inputs - mean) / (variance + self.epsilon)), tf.float32)
            normalized = 0.5 * (tf.nn.tanh(tanh_estimator) + 1.0) * filter
            output = self.gamma_kernel * normalized + self.beta_kernel
        else:
            mean, variance = tf.nn.moments(inputs, [-1], keepdims=True)
            normalized = (inputs - mean) * tf.math.rsqrt(variance + self.epsilon)
            output = self.gamma_kernel * normalized + self.beta_kernel
        return output

    def get_config(self):
        c = {"epsilon": self.epsilon}
        return c


class NormBlock(tf.keras.layers.Layer):
    """Wrapper class that applies layer pre-processing and post-processing.
        pre-processing: x + layer(ln(x)))
        post-processing: ln(x + layer(x))

        NOTE that pre-processing has a better performance on deep models.
    """

    def __init__(self, layer, dropout, pre_mode=True, epsilon=1e-6):
        super(NormBlock, self).__init__()
        self.layer = layer
        self.dropout = dropout
        self.pre_mode = pre_mode
        self.epsilon = epsilon

    def build(self, input_shape):
        self.layer_norm = LayerNorm(epsilon=self.epsilon)
        super(NormBlock, self).build(input_shape)

    def get_config(self):
        return {"dropout": self.dropout, "add_mode": self.add_mode}

    def call(self, x, *args, **kwargs):
        """Calls wrapped layer with same parameters."""
        training = kwargs["training"]
        if self.pre_mode:
            y = self.layer_norm(x)
            y = self.layer(y, *args, **kwargs)
            if training:
                y = tf.nn.dropout(y, self.dropout,)

            y = y + x

        else:
            y = self.layer(x, *args, **kwargs)
            if training:
                y = tf.nn.dropout(y, rate=self.dropout)
            y = self.layer_norm(y + x)
        return y
