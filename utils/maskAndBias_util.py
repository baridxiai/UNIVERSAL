# -*- coding: utf-8 -*-
# code warrior: Barid
import tensorflow as tf
from UNIVERSAL.utils import padding_util


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
