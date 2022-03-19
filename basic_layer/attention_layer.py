# -*- coding: utf-8 -*-
# code warrior: Barid
import tensorflow as tf
import numpy as np
import sys
import math

_NEG_INF_FP32 = -1e9
_NEG_INF_FP16 = np.finfo(np.float16).min


def split_last_dimension(x, n):
    """Reshape x so that the last dimension becomes two dimensions.
  The first of these two dimensions is n.
  Args:
    x: a Tensor with shape [..., m]
    n: an integer.
  Returns:
    a Tensor with shape [..., n, m/n]
  """
    x_shape = tf.shape(x)
    m = x_shape[-1]
    if isinstance(m, int) and isinstance(n, int):
        assert m % n == 0
    return tf.reshape(x, x_shape[:-1] + [n, m // n])


def attention_image_summary(attn, image_shapes=None):
    """Compute color image summary.
  Args:
    attn: a Tensor with shape [batch, num_heads, query_length, memory_length]
    image_shapes: optional tuple of integer scalars.
      If the query positions and memory positions represent the
      pixels of flattened images, then pass in their dimensions:
        (query_rows, query_cols, memory_rows, memory_cols).
      If the query positions and memory positions represent the
      pixels x channels of flattened images, then pass in their dimensions:
        (query_rows, query_cols, query_channels,
         memory_rows, memory_cols, memory_channels).
  """
    attn = tf.cast(attn, tf.float32)
    num_heads = tf.shape(attn)[1]
    # [batch, query_length, memory_length, num_heads]
    image = tf.transpose(attn, [0, 2, 3, 1])
    image = tf.pow(image, 0.2)  # for high-dynamic-range
    # Each head will correspond to one of RGB.
    # pad the heads to be a multiple of 3
    image = tf.pad(image, [[0, 0], [0, 0], [0, 0], [0, tf.math.mod(-num_heads, 3)]])
    image = split_last_dimension(image, 3)
    image = tf.reduce_max(image, 4)
    if image_shapes is not None:
        if len(image_shapes) == 4:
            q_rows, q_cols, m_rows, m_cols = list(image_shapes)
            image = tf.reshape(image, [-1, q_rows, q_cols, m_rows, m_cols, 3])
            image = tf.transpose(image, [0, 1, 3, 2, 4, 5])
            image = tf.reshape(image, [-1, q_rows * m_rows, q_cols * m_cols, 3])
        else:
            assert len(image_shapes) == 6
            q_rows, q_cols, q_channnels, m_rows, m_cols, m_channels = list(image_shapes)
            image = tf.reshape(image, [-1, q_rows, q_cols, q_channnels, m_rows, m_cols, m_channels, 3])
            image = tf.transpose(image, [0, 1, 4, 3, 2, 5, 6, 7])
            image = tf.reshape(image, [-1, q_rows * m_rows * q_channnels, q_cols * m_cols * m_channels, 3])
    tf.summary.image("attention", image, max_outputs=1)


def split_heads(x, heads):
    """Split x into different heads, and transpose the resulting value.
The tensor is transposed to insure the inner dimensions hold the correct
values during the matrix multiplication.
Args:
  x: A tensor with shape [batch_size, length, num_units]
Returns:
  A tensor with shape [batch_size, num_heads, length, num_units/num_heads]
"""
    with tf.name_scope("split_heads"):
        batch_size, length, num_units = tf.unstack(tf.shape(x))
        # Calculate depth of last dimension after it has been split.
        depth = num_units // heads

        # Split the last dimension
        x = tf.reshape(x, [batch_size, length, heads, depth])

        # Transpose the result
        return tf.transpose(x, [0, 2, 1, 3])


def combine_heads(x, num_units):
    """Combine tensor that has been split.
Args:
  x: A tensor [batch_size, num_heads, length, num_units/num_heads]
Returns:
  A tensor with shape [batch_size, length, num_units]
"""
    with tf.name_scope("combine_heads"):
        batch_size, _, length, _ = tf.unstack(tf.shape(x))
        x = tf.transpose(x, [0, 2, 1, 3])  # --> [batch, length, num_heads, depth]
        return tf.reshape(x, [batch_size, length, num_units])


def scaled_dot_product_attention(q, k, v, mask, dropout=0, scale=None):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """

    if scale is None:
        scale = tf.cast(tf.shape(k)[-1], q.dtype) ** -0.5
    else:
        scale = tf.cast(scale, q.dtype)
    mask = tf.cast(mask, q.dtype)
    logits = tf.matmul(q, k, transpose_b=True) * scale
    if mask is not None:
        neg_inf = _NEG_INF_FP16 if logits.dtype == tf.float16 else _NEG_INF_FP32
        logits += mask * neg_inf
    # Note that softmax internally performs math operations using float32
    # for numeric stability. When training with float16, we keep the input
    # and output in float16 for better performance.
    weights = tf.nn.softmax(logits, axis=-1, name="attention_weights")
    if dropout != 0:
        weights = tf.nn.dropout(weights, rate=dropout)
    # else:
    #     attention_image_summary(weights)
    output = tf.matmul(weights, v)
    return output, weights


class Attention(tf.keras.layers.Layer):
    """Multi-headed attention layer."""

    def __init__(self, num_units, num_heads, dropout):
        """Initialize Attention.
    Args:
      num_units: int, output dim of hidden layer.
      num_heads: int, number of heads to repeat the same attention structure.
      attention_dropout: float, dropout rate inside attention for training.
    """
        if num_units % num_heads:
            raise ValueError(
                "Hidden size ({}) must be divisible by the number of heads ({}).".format(num_units, num_heads)
            )

        super(Attention, self).__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.dropout = dropout
        self.attention_weights = 0

    def build(self, input_shape):
        """Builds the layer."""
        # Layers for linearly projecting the queries, keys, and values.
        # def _glorot_initializer(fan_in, fan_out):
        #     limit = math.sqrt(6.0 / (fan_in + fan_out))
        #     return tf.keras.initializers.RandomUniform(minval=-limit, maxval=limit)

        # attention_initializer = _glorot_initializer(input_shape.as_list()[-1], self.num_units)
        self.q_dense_layer = tf.keras.layers.Dense(
            self.num_units, name="q", use_bias=False, kernel_initializer=tf.keras.initializers.glorot_uniform
        )
        self.k_dense_layer = tf.keras.layers.Dense(
            self.num_units, name="k", use_bias=False, kernel_initializer=tf.keras.initializers.glorot_uniform
        )
        self.v_dense_layer = tf.keras.layers.Dense(
            self.num_units, name="v", use_bias=False, kernel_initializer=tf.keras.initializers.glorot_uniform
        )
        self.output_dense_layer = tf.keras.layers.Dense(
            self.num_units,
            name="output_transform",
            use_bias=False,
            kernel_initializer=tf.keras.initializers.glorot_uniform,
        )

        super(Attention, self).build(input_shape)

    def get_config(self):
        return {
            "num_units": self.num_units,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
        }

    def call(self, x, y, bias, training, cache=None, scale=None, **kwargs):
        """Apply attention mechanism to x and y.
    Args:
      x: a tensor with shape [batch_size, length_x, num_units]
      y: a tensor with shape [batch_size, length_y, num_units]
      bias: attention bias that will be added to the result of the dot product.
      training: boolean, whether in training mode or not.
      cache: (Used during prediction) dictionary with tensors containing results
        of previous attentions. The dictionary must have the items:
            {"k": tensor with shape [batch_size, i, key_channels],
             "v": tensor with shape [batch_size, i, value_channels]}
        where i is the current decoded length.
    Returns:
      Attention layer output with shape [batch_size, length_x, num_units]
    """
        # Linearly project the query (q), key (k) and value (v) using different
        # learned projections. This is in preparation of splitting them into
        # multiple heads. Multi-head attention uses multiple queries, keys, and
        # values rather than regular attention (which uses a single q, k, v).
        # padding_bias = tf.expand_dims(
        #     tf.cast(tf.not_equal(tf.reduce_sum(x, -1), 0), tf.float32), -1)
        # if len(x) > 1:
        #     print("two stream attention")
        #     x, x_2nd = x
        q = self.q_dense_layer(x)
        k = self.k_dense_layer(y)
        v = self.v_dense_layer(y)

        q = split_heads(q, self.num_heads)
        k = split_heads(k, self.num_heads)
        v = split_heads(v, self.num_heads)
        bias = tf.expand_dims(bias, 1)  # [X, X,X] --> [X,1,X,X] expand dims for head
        if cache is not None:
            # Combine cached keys and values with new keys and values.
            k = tf.concat((cache["k"], k), axis=2)
            v = tf.concat((cache["v"], v), axis=2)
            # Update cache
            cache["k"] = k
            cache["v"] = v
        if training:
            attention_output, self.attention_weights = scaled_dot_product_attention(q, k, v, bias, self.dropout, scale)
        else:
            attention_output, self.attention_weights = scaled_dot_product_attention(q, k, v, bias, scale=scale)
        attention_output = combine_heads(attention_output, self.num_units)
        attention_output = self.output_dense_layer(attention_output)
        return attention_output

    def get_attention_weights(self):
        return self.attention_weights


class SelfAttention(Attention):
    """Multiheaded self-attention layer."""

    def call(self, x, bias, training, cache=None, **kwargs):
        # if len(x) > 1:
        #     return super(SelfAttention, self).call(x, x[0], bias, training, cache,
        #                                            **kwargs)
        #
        # else:
        return super(SelfAttention, self).call(x, x, bias, training, cache, **kwargs)
