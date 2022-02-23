# -*- coding: utf-8 -*-
# code warrior: Barid
import tensorflow as tf
import numpy as np

def local_perm(inputs, is_masked, perm_size, seq_len, leak_ratio):
    # Generate permutation indices
    index = tf.range(seq_len, dtype=tf.int64)
    index = tf.transpose(tf.reshape(index, [-1, perm_size]))
    index = tf.random.shuffle(index)
    index = tf.reshape(tf.transpose(index), [-1])

    # non-functional tokens
    non_func_tokens = tf.logical_not(
        tf.logical_or(tf.equal(inputs, 1), tf.equal(inputs, 0)))
    masked_tokens = tf.logical_and(is_masked, non_func_tokens)
    non_masked_or_func_tokens = tf.logical_not(masked_tokens)

    smallest_index = -2 * tf.ones([seq_len], dtype=tf.int64)

    # Similar to BERT, randomly leak some masked tokens
    if leak_ratio > 0:
        leak_tokens = tf.logical_and(
            masked_tokens,
            tf.random.uniform([seq_len], maxval=1.0) < leak_ratio)
        can_attend_self = tf.logical_or(non_masked_or_func_tokens, leak_tokens)
    else:
        can_attend_self = non_masked_or_func_tokens
    to_index = tf.where(can_attend_self, smallest_index, index)
    from_index = tf.where(can_attend_self, to_index + 1, to_index)

    # For masked tokens, can attend if i > j
    # For context tokens, always can attend each other
    can_attend = from_index[:, None] > to_index[None, :]

    # In modeling, 1 indicates cannot attend. Hence, reverse the value here.
    perm_mask = 1.0 - tf.cast(can_attend, tf.float32)

    # Only masked tokens are included in the loss
    target_mask = tf.cast(masked_tokens, tf.float32)

    # construct inputs_k
    inputs_k = inputs

    # construct inputs_q
    inputs_q = masked_tokens

    return perm_mask, target_mask, inputs_k, inputs_q

def random_mask(inputs, T_len, perm_size=10, span=10000):
    temp = []
    batch = tf.shape(inputs)[0]
    # mask = tf.constant(mask, shape=[batch])
    seed = np.random.randint(0, T_len, perm_size)
    mask = np.random.randint(0, span)
    for i in range(1, T_len):
        if i in seed:
            temp.append(tf.zeros_like(inputs[:, i]) + mask)
            continue
        else:
            temp.append(inputs[:, i])
    temp = tf.transpose(temp, perm=[1, 0])
    return temp
    
def denoise_mask(inputs, T_len, perm_size=10, mask=2):
    temp = []
    batch = tf.shape(inputs)[0]
    # mask = tf.constant(mask, shape=[batch])
    seed = np.random.randint(0, T_len, perm_size)
    for i in range(1, T_len):
        if i in seed:
            temp.append(tf.zeros_like(inputs[:, i]) + mask)
            continue
        else:
            temp.append(inputs[:, i])
    temp = tf.transpose(temp, perm=[1, 0])
    return temp


def denoise_swap(inputs, T_len, perm_size):
    index = tf.range(T_len, dtype=tf.int64)
    remainder = tf.math.floormod(T_len, perm_size, name=None)
    # if (T_len % perm_size == 0):
    #     index = tf.transpose(tf.reshape(index, [-1, perm_size]))
    #     index = tf.random.shuffle(index)
    #     index = tf.reshape(tf.transpose(index), [-1])
    #
    # else:
    remainder = T_len % perm_size
    temp_index = index[0:T_len - remainder]
    temp_index = tf.transpose(tf.reshape(temp_index, [-1, perm_size]))
    temp_index = tf.random.shuffle(temp_index)
    temp_index = tf.reshape(tf.transpose(temp_index), [-1])
    index = tf.concat([temp_index, index[-remainder:]], 0)
    temp = [inputs[:, index[0]]]
    for i in range(1, T_len):
        # if inputs[:, index[i]] != 0:
        temp.append(inputs[:, index[i]])
    # if len(inputs.get_shape().as_list()) == 2:
    #     temp = tf.transpose(temp, perm=[1, 0])
    # else:
    temp = tf.transpose(temp, perm=[1, 0, 2])

    return tf.concat((temp, inputs[:, T_len:]), 1)


def mask_lm(inputs, mask_length):
    mask_extra = np.random.choice([1., 0.], [1, mask_length, 1], p=[0.8, 0.2])
    return inputs * mask_extra
