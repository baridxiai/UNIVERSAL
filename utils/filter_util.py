# -*- coding: utf-8 -*-
# code warrior: Barid
import numpy as np


def domain_filter(vocabulary, domain_index, reserve_list=None):
    """
        Setting filtered ids to -1e9
        return [vocabulary]
    """
    zero_padding = np.zeros(vocabulary)
    zero_padding[domain_index] = 1
    if reserve_list is not None:
        zero_padding[reserve_list] = 1
    zero_padding = (1 - zero_padding) * -1e9
    # return tf.reshape(zero_padding, [1, 1, -1])
    return zero_padding
