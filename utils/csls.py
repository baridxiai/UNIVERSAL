# -*- coding: utf-8 -*-
# code warrior: Barid
import tensorflow as tf
import numpy as np
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
    xz = tf.matmul(x, z, transpose_b=True)
    norm_x = tf.norm(x, axis=-1, keepdims=True)
    norm_z = tf.norm(z, axis=-1, keepdims=True)
    norm_xz = tf.matmul(norm_x, norm_z, transpose_b=True)
    cos = tf.math.divide_no_nan(xz, norm_xz)
    bi_csls = cos * mask_out
    bi_csls = tf.nn.top_k(bi_csls,k)[0]
    mask_out = tf.nn.top_k(mask_out,k)[0]
    return tf.reduce_sum(bi_csls) / tf.reduce_sum(mask_out)