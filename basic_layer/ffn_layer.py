# -*- coding: utf-8 -*-
# code warrior: Barid
import tensorflow as tf
from UNIVERSAL.utils import padding_util


class Feed_Forward_Network(tf.keras.layers.Layer):
    """Fully connected feedforward network."""

    def __init__(self, num_units, dropout, activation_filter="relu", activation_output=None):
        """Initialize FeedForwardNetwork.
    Args:
      num_units: int, output dim of hidden layer.
      filter_size: int, filter size for the inner (first) dense layer.
      relu_dropout: float, dropout rate for training.
    """
        super(Feed_Forward_Network, self).__init__()
        self.num_units = num_units
        self.dropout = dropout
        self.activation_filter = activation_filter
        self.activation_output = activation_output

    def build(self, input_shape):
        self.out_dim = input_shape[-1]
        self.filter_dense_layer = tf.keras.layers.Dense(
            self.num_units,
            use_bias=True,
            activation=self.activation_filter,
            name="filter_layer",
            kernel_initializer=tf.keras.initializers.glorot_uniform,
        )
        self.output_dense_layer = tf.keras.layers.Dense(
            self.out_dim, use_bias=True, name="output_layer", kernel_initializer=tf.keras.initializers.glorot_uniform,
        )
        super(Feed_Forward_Network, self).build(input_shape)

    def get_config(self):
        return {
            "num_units": self.num_units,
            "dropout": self.dropout,
        }

    def call(self, x, training=False, padding_position=None):
        """Return outputs of the feedforward network.
    Args:
      x: tensor with shape [batch_size, length, num_units]
      training: boolean, whether in training mode or not.
    Returns:
      Output of the feedforward network.
      tensor with shape [batch_size, length, num_units]
    """
        if padding_position is not None:
            x = padding_util.seq_padding_remover(x, padding_position)
        output = self.filter_dense_layer(x)
        if training:
            output = tf.nn.dropout(output, rate=self.dropout)
        output = self.output_dense_layer(output)

        if padding_position is not None:
            output = padding_util.seq_padding_restore(output, padding_position)
        return output
