# -*- coding: utf-8 -*-
# code warrior: Barid
import tensorflow as tf

FILTER_MIN = 0
FILTER_MAX = 80


# def prepare_training_input(
#     dataset,
#     batch_size,
#     max_sequence_length,
#     min_boundary=8,
#     boundary_scale=1.1,
#     filter_min=1,
#     filter_max=80,
#     tf_encode=None,
#     shuffle=1000000,
#     stream=256,
#     **kwargs
# ):
#     def _filter_max(src, tgt):

#         return tf.logical_and(tf.size(src) <= filter_max, tf.size(tgt) <= filter_max)

#     def _filter_min(src, tgt):

#         return tf.logical_and(tf.size(src) >= filter_min, tf.size(tgt) >= filter_min)

#     def _get_example_length(*args):
#         """Returns the maximum length between the example inputs and targets."""
#         # src,tgt = src_tgt
#         # if len(src>1):
#         length = tf.maximum(tf.shape(args[0])[0], tf.shape(args[-3])[0])
#         # else:
#         #     length = tf.maximum(tf.shape(src), tf.shape(tgt))
#         return length

#     def _create_max_boundaries_and_batch_size(
#         tokens, max_length, min_boundary=min_boundary, boundary_scale=boundary_scale
#     ):
#         bucket_boundaries = []
#         x = min_boundary
#         while x <= max_length:
#             bucket_boundaries.append(x)
#             x = max(x + 1, min(int(x * boundary_scale), x + min_boundary))
#         bucket_boundaries[-1] = max_length
#         buckets_max = bucket_boundaries
#         bucket_batch_sizes = [tokens // x for x in bucket_boundaries]
#         bucket_batch_sizes = bucket_batch_sizes + [bucket_batch_sizes[-1]]
#         # bucket_id will be a tensor, so convert this list to a tensor as well.
#         bucket_batch_sizes = tf.constant(bucket_batch_sizes, dtype=tf.int64)
#         return buckets_max, bucket_batch_sizes

#     buckets_max, bucket_batch_sizes = _create_max_boundaries_and_batch_size(
#         batch_size, max_sequence_length
#     )
#     dataset = dataset.filter(_filter_min)
#     dataset = dataset.filter(_filter_max)
#     if tf_encode is None:

#         def tf_encode(src, tgt):
#             return (src, tgt)

#     dataset = dataset.map(tf_encode, num_parallel_calls=tf.data.experimental.AUTOTUNE)

#     dataset = dataset.bucket_by_sequence_length(
#         element_length_func=_get_example_length,
#         bucket_boundaries=buckets_max,
#         bucket_batch_sizes=bucket_batch_sizes,
#     )

#     def tf_output(*args):
#         return (args,)

#     dataset = dataset.map(tf_output, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#     dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
#     return dataset


def greedyBatch_training_input(
    dataset,
    parmeters,
    min_boundary=8,
    boundary_scale=1.1,
    filter_min=1,
    filter_max=80,
    preprocess_fn=None, # e.g., MLM input sentence
    postprocess_fn=None,
    **kwargs
):
    def _filter_max(*args):

        return tf.size(args[0][0]) <= filter_max

    def _filter_min(*args):

        return tf.size(args[0][0]) >= filter_min

    def _get_example_length(*args):
        """
        Returns the maximum length between the example inputs and targets.
        Parall corpora: args: [src,tgt]
        Monolingual Corpora args: [src, label,...]
        """
        # src,tgt = src_tgt
        # if len(src>1):
        length = tf.maximum(tf.shape(args[0])[0], tf.shape(args[1])[0])
        # else:
        #     length = tf.maximum(tf.shape(src), tf.shape(tgt))
        return length

    def _create_max_boundaries_and_batch_size(
        tokens, max_length, min_boundary=min_boundary, boundary_scale=boundary_scale
    ):
        bucket_boundaries = []
        x = min_boundary
        while x <= max_length:
            bucket_boundaries.append(x)
            x = max(x + 1, min(int(x * boundary_scale), x + min_boundary))
        bucket_boundaries[-1] = max_length
        buckets_max = bucket_boundaries
        bucket_batch_sizes = [tokens // x for x in bucket_boundaries]
        bucket_batch_sizes = bucket_batch_sizes + [bucket_batch_sizes[-1]]
        # bucket_id will be a tensor, so convert this list to a tensor as well.
        bucket_batch_sizes = tf.constant(bucket_batch_sizes, dtype=tf.int64)
        return buckets_max, bucket_batch_sizes

    buckets_max, bucket_batch_sizes = _create_max_boundaries_and_batch_size(
        parmeters["batch_size"], parmeters["max_sequence_length"]
    )
    dataset = dataset.filter(_filter_min)
    dataset = dataset.filter(_filter_max)
    if preprocess_fn is None:

        def preprocess_fn(*args):
            return args

    dataset = dataset.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.bucket_by_sequence_length(
        element_length_func=_get_example_length,
        bucket_boundaries=buckets_max,
        bucket_batch_sizes=bucket_batch_sizes,
    )
    if postprocess_fn is None:
        def postprocess_fn(*args):
            return (args,)
    dataset = dataset.map(postprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
