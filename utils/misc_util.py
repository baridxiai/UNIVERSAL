# -*- coding: utf-8 -*-
# code warrior: Barid
import tensorflow as tf
import numpy as np

_MIN_BOUNDARY = 8
_BOUNDARY_SCALE = 1.1


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
def csls_score(x, z, k, inplace=False):  # TODO Assuming that axis is 1
    # xp = get_array_module(m)
    m = z.dot(x.T)
    n = m.shape[0]
    ans = np.zeros(n, dtype=m.dtype)
    if k <= 0:
        return ans
    if not inplace:
        m = np.array(m)
    ind0 = np.arange(n)
    ind1 = np.empty(n, dtype=int)
    minimum = m.min()
    for i in range(k):
        m.argmax(axis=1, out=ind1)
        ans += m[ind0, ind1]
        m[ind0, ind1] = minimum
    csls = 2 * x.dot(z.T) - ans / k
    return csls
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


def tf_csls_top1_sort_z(x, z):
    m = tf.matmul(z, x, transpose_b=True)
    n = tf.matmul(x, z, transpose_b=True)
    m = tf.nn.top_k(m)[0]
    candidate = tf.squeeze(tf.nn.top_k(2 * n - m)[1], -1)
    return candidate


def bi_csls_score_batch(x, z, k=100,
                        mask_out=None):  # TODO Assuming that axis is 1
    if mask_out is None:
        mask_out = tf.expand_dims(
            tf.cast(tf.not_equal(tf.reduce_sum(x, -1), 0), tf.float32), -1)
    import pdb; pdb.set_trace()
    xz = tf.matmul(x, z, transpose_b=True)
    norm_x = tf.norm(x, axis=-1, keepdims=True)
    norm_z = tf.norm(z, axis=-1, keepdims=True)
    norm_xz = tf.matmul(norm_x, norm_z, transpose_b=True)
    cos = tf.math.divide_no_nan(xz, norm_xz)
    bi_csls = cos * mask_out
    bi_csls = tf.nn.top_k(bi_csls,k)[0]
    mask_out = tf.nn.top_k(mask_out,k)[0]
    return tf.reduce_sum(bi_csls) / tf.reduce_sum(mask_out)
