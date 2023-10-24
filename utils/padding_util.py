# -*- coding: utf-8 -*-
# code warrior: Barid
import tensorflow as tf

_NEG_INF = -1e9


def seq_mask_prediction(x, mask, mask_id):
    """Remove padding from the given tensor.
    Args:
      x (tf.Tensor): of shape [dim_origin,...]
    Returns:
      a tensor of shape [dim_compressed,...] with dim_compressed <= dim_origin
    """
    # mask = tf.squeeze(mask,[1,2])
    # x += tf.expand_dims(mask,-1)
    with tf.name_scope("mask_reduce/remove"):
        # x_shape = x.get_shape().as_list()
        x = tf.gather_nd(x, indices=tf.cast(tf.where(mask == mask_id), tf.int32),)
        # if not tf.executing_eagerly():
        # This is a hack but for some reason, gather_nd return a tensor of
        # undefined shape, so the shape is set up manually
        # x.set_shape([None] + x_shape[1:])
    return x


def seq_padding_remover(x, pad_mask):
    """Remove padding from the given tensor.
    Args:
      x (tf.Tensor): of shape [dim_origin,...]
    Returns:
      a tensor of shape [dim_compressed,...] with dim_compressed <= dim_origin
    """
    with tf.name_scope("pad_reduce/remove"):
        # x_shape = x.get_shape().as_list()
        x = tf.gather_nd(x, indices=tf.cast(tf.where(pad_mask == 0), tf.int32),)
        # if not tf.executing_eagerly():
        # This is a hack but for some reason, gather_nd return a tensor of
        # undefined shape, so the shape is set up manually
        # x.set_shape([None] + x_shape[1:])
    return x


def seq_padding_restore(x, pad_mask):
    """Add padding back to the given tensor.
    Args:
      x (tf.Tensor): of shape [dim_compressed,...]
    Returns:
      a tensor of shape [dim_origin,...] with dim_compressed >= dim_origin. The
      dim is restored from the original reference tensor
    """
    # pad_mask = tf.expand_dims(pad_mask,-1)
    with tf.name_scope("pad_reduce/restore"):
        x = tf.scatter_nd(
            indices=tf.cast(tf.where(pad_mask == 0), tf.int32),
            updates=x,
            shape=tf.concat([tf.shape(pad_mask), tf.shape(x)[1:]], axis=0),
        )
    # x += tf.expand_dims(pad_mask,-1)
    return x


def get_padding_bias(x):
    """Calculate bias tensor from padding values in tensor.
  Bias tensor that is added to the pre-softmax multi-headed attention logits,
  which has shape [batch_size, num_heads, length, length]. The tensor is zero at
  non-padding locations, and -1e9 (negative infinity) at padding locations.
  Args:
    x: int tensor with shape [batch_size, length]
  Returns:
    Attention bias tensor of shape [batch_size, 1, length].
  """
    with tf.name_scope("attention_bias"):
        padding = get_padding(x)
        attention_bias = padding
        attention_bias = tf.expand_dims(attention_bias, axis=1)

    return attention_bias


def get_embedding_padding(x, padding_value=0):
    padding = tf.cast(tf.not_equal(x, padding_value), tf.float32)
    padding = tf.reduce_mean(padding,axis=-1)
    return padding


def get_padding(x, padding_value=0):
    """Return float tensor representing the padding values in x.
  Args:
    x: int tensor with any shape
    padding_value: int value that
  Returns:
    float tensor with same shape as x containing values 0 or 1.
      0 -> non-padding, 1 -> padding
  """
    with tf.name_scope("padding"):
        return tf.cast(tf.equal(x, padding_value), tf.float32)


def pad_tensors_to_same_length(x, y, pad_id=0):
    """Pad x and y so that the results have the same length (second dimension)."""
    x_length = tf.shape(input=x)[1]
    y_length = tf.shape(input=y)[1]

    max_length = tf.maximum(x_length, y_length)
    if len(x.get_shape().as_list()) == 3:
        x = tf.pad(
            tensor=x, paddings=[[0, 0], [0, max_length - x_length], [0, 0]], constant_values=pad_id,
        )
    else:

        x = tf.pad(tensor=x, paddings=[[0, 0], [0, max_length - x_length]], constant_values=pad_id,)
    if len(y.get_shape().as_list()) == 3:
        y = tf.pad(
            tensor=y, paddings=[[0, 0], [0, max_length - y_length], [0, 0]], constant_values=pad_id,
        )
    else:

        y = tf.pad(tensor=y, paddings=[[0, 0], [0, max_length - y_length]], constant_values=pad_id,)
    return x, y


def pad_list_tensors_to_same_length(x_list, pad_id=0):
    """Pad x and y so that the results have the same length (second dimension)."""
    x_length_can = [tf.shape(input=x)[0] for x in x_list]

    max_length = tf.reduce_max(x_length_can)
    x_list = [
        tf.pad(tensor=x, paddings=[[0, max_length - tf.shape(input=x)[0]]], constant_values=pad_id,)
        for x in x_list
    ]
    return x_list



def get_decoder_self_attention_bias(length, x=None, lower=-1, upper=0):
    """Calculate bias for decoder that maintains model's autoregressive property.
  Creates a tensor that masks out locations that correspond to illegal
  connections, so prediction at position i cannot draw information from future
  positions.
  Args:
    length: int length of sequences in batch.
  Returns:
    float tensor of shape [1, length, length]
  """
    if x is not None:
        length = tf.shape(x)[1]
        padding_mask = padding_util.get_padding(x)
        padding_mask = tf.expand_dims(padding_mask, axis=1)

    with tf.name_scope("decoder_self_attention_bias"):
        valid_locs = tf.linalg.band_part(tf.ones([length, length]), lower, upper)
        valid_locs = tf.reshape(valid_locs, [1, length, length])
    decoder_bias = 1.0 - valid_locs
    if x is not None:
        decoder_bias = tf.maximum(padding_mask, decoder_bias)
        # valid_locs = tf.cast(tf.greater(valid_locs, 0), tf.float32)
        # tf.maximum(dec_target_padding_mask, look_ahead_mask)
    return decoder_bias