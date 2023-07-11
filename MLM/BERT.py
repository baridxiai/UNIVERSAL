# -*- coding: utf-8 -*-
# code warrior: Barid
import tensorflow as tf


def BERT_masking(
    input_ids,
    vocabulary_size,
    all_special_ids=[0],
    masking_id=4,
    mlm_probability=0.15,
    mlm_ratio=[0.8, 0.1, 0.1],
    label_nonmasking=0,
):
    """_summary_

    Args:
        input_ids : integer sequence
        all_special_id : non-masing ids like padding, SOS, EOS
        masking_id : mask id Defaults to 4.
        mlm_probability: how many tokens to be masked Defaults to 0.15.
        mlm_ration: [MASK,  random , unchange] Defaults to [0.8,0.1,0.1].
        label_nonmasking: mark unpredictable token

    Returns:
        masked_inputs, labels
    """
    probability_matrix = tf.fill(tf.shape(input_ids), mlm_probability)

    org = input_ids
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
    probability_matrix = tf.where(mask_ignores, 0.0, probability_matrix)
    masked_indices = tf.cast(tf.keras.backend.random_bernoulli(tf.shape(input_ids), probability_matrix), tf.bool)

    # labels[~masked_indices] = -1  # We only compute loss on masked tokens
    labels = tf.where(masked_indices, input_ids, label_nonmasking)

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = (
        tf.cast(
            tf.keras.backend.random_bernoulli(tf.shape(input_ids), tf.fill(tf.shape(input_ids), mlm_ratio[0])), tf.bool
        )
        & masked_indices
    )
    input_ids = tf.where(indices_replaced, masking_id, input_ids)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = (
        tf.cast(
            tf.keras.backend.random_bernoulli(
                tf.shape(input_ids), tf.fill(tf.shape(input_ids), mlm_ratio[1] / (mlm_ratio[1] + mlm_ratio[2]))
            ),
            tf.bool,
        )
        & masked_indices
        & ~indices_replaced
    )

    # random_words = np.random.randint(len(tokenizer), size=labels.shape, dtype=np.int32)
    random_words = tf.random.uniform(shape=tf.shape(input_ids), maxval=vocabulary_size, dtype=tf.int32)
    input_ids = tf.where(indices_random, random_words, input_ids)

    return input_ids, org ,masked_indices , labels
