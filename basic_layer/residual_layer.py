# -*- coding: utf-8 -*-
# code warrior: Barid
import tensorflow as tf


class NaiveRes(tf.keras.layers.Layer):
    """Wrapper class that applies residual connection.

        return x+ dropout(layer(x))
    """
    def __init__(self, layer, dropout, pre_mode=True):
        super(NaiveRes, self).__init__()
        self.layer = layer
        self.dropout = dropout

    def get_config(self):
        return {"dropout": self.dropout}

    def call(self, x, *args, **kwargs):
        """Calls wrapped layer with same parameters."""
        # Preprocessing: apply layer normalization
        training = kwargs["training"]
        y = self.layer(x, *args, **kwargs)
        if training:
            #     y = tf.nn.dropout(y, rate=self.dropout)
            y_shape = tf.shape(y)
            y = tf.nn.dropout(y,
                              self.dropout,
                              noise_shape=[y_shape[0], 1, y_shape[2]])

        y = y + x

        return y
