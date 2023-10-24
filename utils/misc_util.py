# -*- coding: utf-8 -*-
# code warrior: Barid
import tensorflow as tf
import numpy as np


def shape_list(x):
    """Return list of dims, statically where possible."""
    x = tf.convert_to_tensor(x)

    # If unknown rank, return dynamic shape
    if x.get_shape().dims is None:
        return tf.shape(x)

    static = x.get_shape().as_list()
    shape = tf.shape(x)

    ret = []
    for i, dim in enumerate(static):
        if dim is None:
            dim = shape[i]
        ret.append(dim)
    return ret


def log_prob_from_logits(logits, reduce_axis=-1):
    return logits - tf.reduce_logsumexp(
        logits, axis=reduce_axis, keepdims=True)



def set_TRIM_ID(id):
    global TRIM_ID
    TRIM_ID = id


def token_trim(tokens, trim_id=0, remider=0):
    try:
        trim = tokens.index(trim_id) + int(remider)
        if trim == 0:
            tokens = tokens[:1]
        else:
            tokens = tokens[:trim]
    except Exception:
        tokens
    return tokens



def gather_indexes(sequence_tensor, positions,twoD=False):
    batch_size = tf.shape(sequence_tensor)[0]
    seq_length = tf.shape(sequence_tensor)[1]

    flat_offsets = tf.reshape(
        tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    if twoD:
        # width = tf.shape(sequence_tensor)[2]
        flat_sequence_tensor = tf.reshape(sequence_tensor,
                                      [batch_size * seq_length])
        output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
        return tf.reshape(output_tensor, [batch_size, -1])
    else:
        width = tf.shape(sequence_tensor)[2]
        flat_sequence_tensor = tf.reshape(sequence_tensor,
                                      [batch_size * seq_length, width])
        output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
        return tf.reshape(output_tensor, [batch_size, -1, width])
def tensor_remove_0(x):
    intermediate_tensor = tf.reduce_sum(tf.abs(x), -1)
    zero_vector = tf.zeros(shape=x.get_shape(), dtype=x.dtype)
    bool_mask = tf.not_equal(intermediate_tensor, zero_vector)
    omit_zeros = tf.boolean_mask(x, bool_mask)
    return omit_zeros


def tensor_reserve_value(x, value=1):
    # intermediate_tensor = tf.reduce_sum(tf.abs(x), 1)
    # zero_vector = tf.zeros(shape=x.get_shape(), dtype=x.dtype)
    # intermediate_tensor = t
    value = tf.cast(value, x.dtype) + tf.zeros(shape=x.get_shape(),
                                               dtype=x.dtype)
    bool_mask = tf.cast(tf.equal(x, value), tf.float32)
    re = x * bool_mask
    return re
