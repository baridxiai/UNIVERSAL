# -*- coding: utf-8 -*-
# code warrior: Barid
import tensorflow as tf
from UNIVERSAL.utils import filter_util as fn_filter


class EmbeddingSharedWeights(tf.keras.layers.Layer):
    """Calculates input embeddings and pre-softmax linear with shared weights."""

    def __init__(
        self, vocab_size, num_units, pad_id=0, scale_we=True, affine=False, name="embedding", domain_index=[],
    ):
        """Specify characteristic parameters of embedding layer.
    Args:
      vocab_size: Number of tokens in the embedding. (Typically ~32,000)
      num_units: Dimensionality of the embedding. (Typically 512 or 1024)
      method: Strategy for performing embedding lookup. "gather" uses tf.gather
        which performs well on CPUs and GPUs, but very poorly on TPUs. "matmul"
        one-hot encodes the indicies and formulates the embedding as a sparse
        matrix multiplication. The matmul formulation is wasteful as it does
        extra work, however matrix multiplication is very fast on TPUs which
        makes "matmul" considerably faster than "gather" on TPUs.


        NOTE that domain_index is used to specify the domain when generating probabilities with _linear().
    """
        super(EmbeddingSharedWeights, self).__init__(name=name)
        self.vocab_size = vocab_size
        self.num_units = num_units
        self.pad_id = pad_id
        self.affine = affine
        self.scale_we = scale_we
        self.shared_weights = self.add_weight(
            shape=[self.vocab_size, self.num_units],
            dtype="float32",
            name="shared_weights",
            # initializer=tf.random_normal_initializer(mean=0.0, stddev=self.num_units ** -0.5),
            initializer=tf.keras.initializers.glorot_uniform,
        )
        self.domain_index = domain_index

    def build(self, input_shape):
        # self.shared_weights = self.em_shared_weights.get_weights()[0]
        if self.affine:
            self.affine_transformation = self.add_weight(
                shape=[self.vocab_size],
                dtype="float32",
                name="shared_weights_affline",
                initializer=tf.random_normal_initializer(mean=0.0, stddev=self.num_units ** -0.5),
            )

        super(EmbeddingSharedWeights, self).build(input_shape)
        # self.build = True

    def call(self, inputs, linear=False, domain_id=None, keep_dim=True):
        if linear:
            return self._linear(inputs, domain_id=domain_id, keep_dim=keep_dim)
        return self._embedding(inputs)

    def _embedding(self, inputs, domain_id=None):
        embeddings = tf.gather(self.shared_weights, inputs)
        mask = tf.cast(tf.not_equal(inputs, 0), embeddings.dtype)
        embeddings *= tf.expand_dims(mask, -1)
        # # Scale embedding by the sqrt of the hidden size
        if self.scale_we:
            embeddings *= self.num_units ** 0.5
        return embeddings

    def _linear(self, inputs, domain_id=None, keep_dim=True):
        """Computes logits by running x through a linear layer.
    Args:
      x: A float32 tensor with shape [batch_size, length, num_units]
    Returns:
      float32 tensor with shape [batch_size, length, vocab_size].
    """
        batch_size = tf.shape(input=inputs)[0]
        length = tf.shape(input=inputs)[1]
        inputs = tf.reshape(inputs, [-1, self.num_units])
        logits = tf.matmul(tf.cast(inputs, self.shared_weights.dtype), self.shared_weights, transpose_b=True)

        return tf.reshape(logits, [batch_size, length, self.vocab_size])

    # def _linear(self, inputs, domain_id=None, keep_dim=True):
    #     """Computes logits by running x through a linear layer.
    # Args:
    #   x: A float32 tensor with shape [batch_size, length, num_units]
    # Returns:
    #   float32 tensor with shape [batch_size, length, vocab_size].
    # """

    #     # logits = tf.matmul(logits, self.projection_weights)
    #     #
    #     def __out_weight(id):
    #         print("Domain filtering: " + str(id))
    #         # domain_id = tf.cast(id,tf.int8)
    #         out_weights = fn_filter.domain_filter(
    #             self.vocab_size, self.domain_index[id]
    #         )
    #         out_weights = tf.cast(out_weights, tf.float32)
    #         return out_weights

    #     batch_size = tf.shape(input=inputs)[0]
    #     length = tf.shape(input=inputs)[1]
    #     logits = tf.reshape(inputs, [-1, self.num_units])

    #     logits = tf.matmul(logits, self.shared_weights, transpose_b=True)
    #     if self.affine:
    #         logits = tf.add(logits, self.affine_transformation)
    #     if domain_id is not None and len(self.domain_index) > 0:
    #         out_weights = tf.py_function(__out_weight, [domain_id], [tf.float32])
    #         # logits = logits + tf.expand_dims(out_weights, 0)
    #         logits = logits + out_weights
    #     if keep_dim:
    #         re = tf.reshape(logits, [batch_size, length, self.vocab_size])
    #         return re
    #     else:
    #         return logits

    def get_config(self):
        # config = super(EmbeddingSharedWeights, self).get_config()
        c = {
            "vocab_size": self.vocab_size,
            "num_units": self.num_units,
            "pad_id": self.pad_id,
            "name": self.name,
            "domain_index": self.domain_index,
            "affine": self.affine,
        }
        # config.update(c)
        return c
