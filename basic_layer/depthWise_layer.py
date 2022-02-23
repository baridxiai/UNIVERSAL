# -*- coding: utf-8 -*-
# code warrior: Barid
import tensorflow as tf


class Depth_Wise_Network(tf.keras.layers.Layer):
    """Fully connected feedforward network."""

    def __init__(
        self, num_units, dropout, activation_filter="gelu", activation_output=None
    ):
        """Initialize FeedForwardNetwork.
    Args:
      num_units: int, output dim of hidden layer.
      filter_size: int, filter size for the inner (first) dense layer.
      relu_dropout: float, dropout rate for training.

    """
        super(Depth_Wise_Network, self).__init__()
        self.num_units = num_units
        self.dropout = dropout
        self.activation_filter = activation_filter
        self.activation_output = activation_output

    def build(self, input_shape):
        out_dim = input_shape[-1]
        self.filter_conv1d_layer = tf.keras.layers.Conv1D(
            filters=self.num_units,
            kernel_size=1,
            strides=1,
            use_bias=True,
            activation=self.activation_filter,
            name="filter_layer",
        )
        self.output_conv1d_layer = tf.keras.layers.Conv1D(
            filters=out_dim,
            kernel_size=1,
            strides=1,
            use_bias=True,
            activation=self.activation_output,
            name="filter_layer",
        )
        # self.mask = tf.keras.layers.Masking(0)
        super(Depth_Wise_Network, self).build(input_shape)

    def get_config(self):
        return {
            "num_units": self.num_units,
            "dropout": self.dropout,
        }

    def call(self, x, training):
        """Return outputs of the feedforward network.
    Args:
      x: tensor with shape [batch_size, length, num_units]
      training: boolean, whether in training mode or not.
    Returns:
      Output of the feedforward network.
      tensor with shape [batch_size, length, num_units]
    """
        # Retrieve dynamically known shapes
        # batch_size = tf.shape(x)[0]
        # length = tf.shape(x)[1]
        output = self.filter_conv1d_layer(x)
        if training:
            output = tf.nn.dropout(output, rate=self.dropout)
        output = self.output_conv1d_layer(output)

        return output
