# -*- coding: utf-8 -*-
# code warrior: Barid
import numpy as np
import tensorflow as tf
from UNIVERSAL.utils import padding_util


def gram_linear(x):
    """Compute Gram (kernel) matrix for a linear kernel.

    Args:
      x: A num_examples x num_features matrix of features.

    Returns:
      A num_examples x num_examples Gram matrix of examples.
    """
    # return x.dot(x.T)
    return x.dot(x.T)


def gram_rbf(x, threshold=1.0):
    """Compute Gram (kernel) matrix for an RBF kernel.

    Args:
      x: A num_examples x num_features matrix of features.
      threshold: Fraction of median Euclidean distance to use as RBF kernel
        bandwidth. (This is the heuristic we use in the paper. There are other
        possible ways to set the bandwidth; we didn't try them.)

    Returns:
      A num_examples x num_examples Gram matrix of examples.
    """
    dot_products = x.dot(x.T)
    sq_norms = np.diag(dot_products)
    sq_distances = -2 * dot_products + sq_norms[:, None] + sq_norms[None, :]
    sq_median_distance = np.median(sq_distances)
    return np.exp(-sq_distances / (2 * threshold ** 2 * sq_median_distance))


def center_gram(gram, unbiased=False):
    """Center a symmetric Gram matrix.

    This is equvialent to centering the (possibly infinite-dimensional) features
    induced by the kernel before computing the Gram matrix.

    Args:
      gram: A num_examples x num_examples symmetric matrix.
      unbiased: Whether to adjust the Gram matrix in order to compute an unbiased
        estimate of HSIC. Note that this estimator may be negative.

    Returns:
      A symmetric matrix with centered columns and rows.
    """
    if not np.allclose(gram, gram.T):
        raise ValueError("Input must be a symmetric matrix.")
    gram = gram.copy()

    if unbiased:
        # This formulation of the U-statistic, from Szekely, G. J., & Rizzo, M.
        # L. (2014). Partial distance correlation with methods for dissimilarities.
        # The Annals of Statistics, 42(6), 2382-2412, seems to be more numerically
        # stable than the alternative from Song et al. (2007).
        n = gram.shape[0]
        np.fill_diagonal(gram, 0)
        means = np.sum(gram, 0, dtype=np.float64) / (n - 2)
        means -= np.sum(means) / (2 * (n - 1))
        gram -= means[:, None]
        gram -= means[None, :]
        np.fill_diagonal(gram, 0)
    else:
        means = np.mean(gram, 0, dtype=np.float64)
        means -= np.mean(means) / 2
        gram -= means[:, None]
        gram -= means[None, :]

    return gram


def cka(gram_x, gram_y, debiased=False):
    """Compute CKA.

    Args:
      gram_x: A num_examples x num_examples Gram matrix.
      gram_y: A num_examples x num_examples Gram matrix.
      debiased: Use unbiased estimator of HSIC. CKA may still be biased.

    Returns:
      The value of CKA between X and Y.
    """
    gram_x = center_gram(gram_x, unbiased=debiased)
    gram_y = center_gram(gram_y, unbiased=debiased)

    # Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant) or
    # n*(n-3) (unbiased variant), but this cancels for CKA.
    scaled_hsic = gram_x.ravel().dot(gram_y.ravel())

    normalization_x = np.linalg.norm(gram_x)
    normalization_y = np.linalg.norm(gram_y)
    return scaled_hsic / (normalization_x * normalization_y)


def _debiased_dot_product_similarity_helper(
    xty, sum_squared_rows_x, sum_squared_rows_y, squared_norm_x, squared_norm_y, n
):
    """Helper for computing debiased dot product similarity (i.e. linear HSIC)."""
    # This formula can be derived by manipulating the unbiased estimator from
    # Song et al. (2007).
    return (
        xty
        - n / (n - 2.0) * sum_squared_rows_x.dot(sum_squared_rows_y)
        + squared_norm_x * squared_norm_y / ((n - 1) * (n - 2))
    )


def feature_space_linear_cka(features_x, features_y, sentence_level=False):
    """Compute CKA with a linear kernel, in feature space.

    This is typically faster than computing the Gram matrix when there are fewer
    features than examples.

    Args:
      features_x: A num_examples x num_features matrix of features.
      features_y: A num_examples x num_features matrix of features.
      debiased: Use unbiased estimator of dot product similarity. CKA may still be
        biased. Note that this estimator may be negative.

    Returns:
      The value of CKA between X and Y.
    """
    # features_x = features_x - np.mean(features_x, 0, keepdims=True)
    # import pdb;pdb.set_trace()
    # features_x = features_x - tf.reduce_mean(features_x, -1, keepdims=True)
    # features_y = features_y - np.mean(features_y, 0, keepdims=True)
    # features_y = features_y - tf.reduce_mean(features_y, -1, keepdims=True)

    # dot_product_similarity = np.linalg.norm(features_x.T.dot(features_y)) ** 2
    if sentence_level:
        # normalization_x = np.linalg.norm(features_x.T.dot(features_x))
        # normalization_y = np.linalg.norm(features_y.T.dot(features_y))

        # dot_product_similarity = tf.norm(tf.matmul(features_x, features_y, transpose_a=True),) ** 2
        # normalization_x = tf.norm(tf.matmul(features_x, features_x, transpose_a=True),)
        # normalization_y = tf.norm(tf.matmul(features_y, features_y, transpose_a=True),)
        features_x = tf.reduce_mean(features_x, -2)
        features_y = tf.reduce_mean(features_y, -2)
    # else:
    dot_product_similarity = (
        tf.norm(tf.matmul(tf.expand_dims(features_x, -1), tf.expand_dims(features_y, -1), transpose_a=True), axis=-1)
        ** 2
    )
    # normalization_x = np.linalg.norm(features_x.T.dot(features_x))
    normalization_x = tf.norm(
        tf.matmul(tf.expand_dims(features_x, -1), tf.expand_dims(features_x, -1), transpose_a=True), axis=-1
    )

    # normalization_y = np.linalg.norm(features_y.T.dot(features_y))
    normalization_y = tf.norm(
        tf.matmul(tf.expand_dims(features_y, -1), tf.expand_dims(features_y, -1), transpose_a=True), axis=-1
    )
    # dot_product_similarity = (
    #     tf.norm(
    #         tf.matmul(tf.expand_dims(features_x, -2), tf.expand_dims(features_y, -2), transpose_a=True),
    #         axis=[-1, -2],
    #     )
    #     ** 2
    # )
    # # normalization_x = np.linalg.norm(features_x.T.dot(features_x))
    # normalization_x = tf.norm(
    #     tf.matmul(tf.expand_dims(features_x, -2), tf.expand_dims(features_x, -2), transpose_a=True), axis=[-1, -2]
    # )

    # # normalization_y = np.linalg.norm(features_y.T.dot(features_y))
    # normalization_y = tf.norm(
    #     tf.matmul(tf.expand_dims(features_y, -2), tf.expand_dims(features_y, -2), transpose_a=True), axis=[-1, -2]
    # )

    # * 512**-0.5
    # return tf.expand_dims(tf.math.divide_no_nan(dot_product_similarity, normalization_x * normalization_y), -1)
    return tf.math.divide_no_nan(dot_product_similarity, normalization_x * normalization_y)


def feature_space_linear_cka_3d_self(x):
    features_x = features_y = x
    dot_product_similarity = tf.matmul(features_x, features_y, transpose_a=True) ** 2
    # normalization_x = np.linalg.norm(features_x.T.dot(features_x))
    # b,s,6,d
    features_x = tf.expand_dims(tf.transpose(features_x, [0, 1, 3, 2]), -1)
    normalization_x = tf.norm(tf.matmul(features_x, features_x, transpose_a=True), axis=-1)
    # normalization_x = tf.expand_dims(tf.norm(features_x**2,axis=-1),-2)**2
    # normalization_y = np.linalg.norm(features_y.T.dot(features_y))
    # features_y = tf.expand_dims(features_y,-1)
    # normalization_y = tf.norm(tf.matmul(features_y,
    #                                           features_x,transpose_a=True),axis=-2)
    normalization_y = tf.transpose(normalization_x, [0, 1, 3, 2])
    re = tf.math.divide_no_nan(dot_product_similarity, normalization_x * normalization_y)
    mask = tf.shape(re)[-1]
    # mask = tf.reshape(tf.eye(mask,mask),[1,1,mask,mask])
    mask = padding_util.get_decoder_self_attention_bias(mask)

    # * 512**-0.5

    return re * mask
