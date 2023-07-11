# -*- coding: utf-8 -*-
# code warrior: Barid
import tensorflow as tf
import functools
from UNIVERSAL.model import MLM_base



def MASS_masking(
    input_ids,
    vocabulary_size=10000,
    span_n=0.5,
    all_special_ids=[0],
    masking_id=4,
    mlm_probability=1.0,
    mlm_ratio=[0.8, 0.1, 0.1],
    label_nonmasking=0,
    batch=False,
):
    """_summary_

    Args:
        input_ids : integer sequence
        all_special_id : non-masing ids like padding, SOS, EOS
        span_n: segment length
        masking_id : mask id Defaults to 4.
        mlm_probability: how many tokens to be masked Defaults to 1.
        mlm_ration: [MASK,  random , unchange] Defaults to [0.8,0.1,0.1].
        label_nonmasking: mark unpredictable token

    Returns:
        masked_inputs, labels
    """
    if batch:
        input_shape = tf.shape(input_ids)
        _, length = tf.unstack(input_shape)
    else:
        length = tf.size(input_ids)
        input_shape = [length]
    length = tf.cast(length, tf.float32)
    span = tf.cast(length * span_n, tf.int32)
    input_span = tf.sequence_mask(span, length, dtype=tf.float32)  # [1,1,1,0,0]
    output_span = tf.sequence_mask(tf.math.subtract(span, 1), length, dtype=tf.float32)  # [1,1,0,0,0]
    shift_seed = tf.random.uniform([1], maxval=span + 1, dtype=tf.int32)
    input_span = tf.roll(input_span, shift_seed, [0])  # [0,0,1,1,1]
    output_span = tf.roll(output_span, shift_seed, [0])  # [0,0,1,1,0]
    probability_matrix = tf.fill(input_shape, mlm_probability)
    mask_ignores = tf.cast(
        tf.reduce_max(
            tf.cast(
                tf.stack(list(input_ids == all_special_ids[i] for i in range(len(all_special_ids))), axis=0,),
                dtype=tf.int32,
            ),
            axis=0,
        ),
        tf.bool,
    )
    mask_ignores_input = mask_ignores
    mask_ignores_output = mask_ignores
    probability_matrix_input = tf.where(mask_ignores_input, 0.0, probability_matrix * input_span)
    probability_matrix_output = tf.where(mask_ignores_output, 0.0, probability_matrix * output_span)
    masked_indices_input = tf.cast(tf.keras.backend.random_bernoulli(input_shape, probability_matrix_input), tf.bool)
    masked_indices_output = tf.cast(tf.keras.backend.random_bernoulli(input_shape, probability_matrix_output), tf.bool)

    # labels[~masked_indices] = -1  # We only compute loss on masked tokens
    labels = tf.where(masked_indices_input, input_ids, label_nonmasking)

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced_input = (
        tf.cast(tf.keras.backend.random_bernoulli(input_shape, tf.fill(input_shape, mlm_ratio[0])), tf.bool)
        & masked_indices_input
    )
    indices_replaced_output = (
        tf.cast(tf.keras.backend.random_bernoulli(input_shape, tf.fill(input_shape, mlm_ratio[0])), tf.bool)
        & masked_indices_output
    )
    MASS_input = tf.where(indices_replaced_input, masking_id, input_ids)
    MASS_output = tf.where(indices_replaced_output, masking_id, input_ids)

    # 10% of the time, we replace masked input tokens with random word
    indices_random_input = (
        tf.cast(
            tf.keras.backend.random_bernoulli(
                input_shape, tf.fill(input_shape, mlm_ratio[1] / (mlm_ratio[1] + mlm_ratio[2]))
            ),
            tf.bool,
        )
        & masked_indices_input
        & ~indices_replaced_input
    )
    indices_random_output = (
        tf.cast(
            tf.keras.backend.random_bernoulli(
                input_shape, tf.fill(input_shape, mlm_ratio[1] / (mlm_ratio[1] + mlm_ratio[2]))
            ),
            tf.bool,
        )
        & masked_indices_output
        & ~indices_replaced_output
    )

    # random_words = np.random.randint(len(tokenizer), size=labels.shape, dtype=np.int32)
    random_words = tf.random.uniform(shape=input_shape, maxval=vocabulary_size, dtype=tf.int32)
    MASS_input = tf.where(indices_random_input, random_words, MASS_input)
    MASS_output = tf.where(indices_random_output, random_words, MASS_output)

    return MASS_input, MASS_output, input_span, labels


def map_wrapper_MASS_masking(
    vocabulary_size=10000,
    span_n=0.3,
    all_special_ids=[0],
    masking_id=4,
    mlm_probability=1.0,
    mlm_ratio=[0.8, 0.1, 0.1],
    label_nonmasking=0,
    batch=False,
):
    wrapper_fn = functools.partial(
        MASS_masking,
        vocabulary_size=vocabulary_size,
        span_n=span_n,
        all_special_ids=all_special_ids,
        masking_id=masking_id,
        mlm_probability=mlm_probability,
        mlm_ratio=mlm_ratio,
        label_nonmasking=label_nonmasking,
        batch=batch,
    )
    return wrapper_fn

class MASS(MLM_base.MLM_base):
    def __init__(self, param, **kwargs):
        super().__init__(param, **kwargs)

    def pre_training(self, data):
        ((input_src, output_tgt, span,tgt_label,lang_ids),) = data
        src_lang_ids =  tgt_lang_ids = lang_ids
        metric = tf.where(tf.equal(input_src, self.param["MASK_ID"]), tgt_label, input_src)
        _ = self.seq2seq_training(
            self.call,
            input_src,
            output_tgt,
            sos=self.param["EOS_ID"],
            src_id=src_lang_ids,
            tgt_id=tgt_lang_ids,
            tgt_label=tgt_label,
            tgt_metric=metric,
            span=span,
        )