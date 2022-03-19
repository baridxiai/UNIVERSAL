# -*- coding: utf-8 -*-
# code warrior: Barid
import tensorflow as tf
from UNIVERSAL.utils import staticEmbedding_util, padding_util
from UNIVERSAL.block import TransformerBlock
from UNIVERSAL.basic_layer import layerNormalization_layer, attention_layer, residual_layer, ffn_layer
import math

# import numpy as np


def input_preprocessing(x, non_padding=None):
    """
    apply non_paddings to x.
    non_padding = [1,1,1,1,0,0,0]
    x  = x * non_paddin

    """
    if non_padding is not None:
        x *= non_padding
    return x


def ut_input_preprocess(
    inputs, step, training=False, position_index=None, step_encoding=True, position_encoding=True, **kwargs
):
    # em_padding = padding_util.get_embedding_padding(inputs)
    # obseving the model could not understand the distinguish Between
    # step position and position encoding becasu they have the same value.
    if "max_step" in kwargs:
        max_step = kwargs["max_step"]
    else:
        max_step = 50
    if "max_seq" in kwargs:
        max_seq = kwargs["max_seq"]
    else:
        max_seq = 1000
    if position_index is not None:
        length = max_seq
    else:
        length = None
    if step_encoding:
        inputs = staticEmbedding_util.add_step_timing_signal(inputs, step, max_step)
    if position_encoding:
        inputs = staticEmbedding_util.add_position_timing_signal(inputs, 0, position=position_index, length=length)
    return inputs


class UniversalTransformerEncoderBLOCK(TransformerBlock.TransformerEncoderBLOCK):
    def call(
        self,
        inputs,
        attention_bias,
        encoder_padding=None,
        enc_position=None,
        training=False,
        step=None,
        position_encoding=True,
        step_encoding=True,
        **kwargs
    ):
        with tf.name_scope("UT_encoder"):
            if step_encoding:
                assert step is not None, "step is required when setting step_encoding = True"
            if encoder_padding is not None:
                input_padding = 1 - tf.expand_dims(encoder_padding, -1)
            else:
                input_padding = None
            inputs = input_preprocessing(inputs, input_padding)
            inputs = ut_input_preprocess(
                inputs,
                step,
                position_index=enc_position,
                position_encoding=position_encoding,
                step_encoding=step_encoding,
                **kwargs
            )
            inputs = super(UniversalTransformerEncoderBLOCK, self).call(
                inputs,
                attention_bias,
                training=training,
                index=step,
                encoder_padding=encoder_padding,
                scale=self.num_units ** -0.5,
                **kwargs
            )
            return inputs


class UniversalTransformerDecoderBLOCK(TransformerBlock.TransformerDecoderBLOCK):
    def call(
        self,
        inputs,
        enc,
        decoder_self_attention_bias,
        attention_bias,
        dec_position=None,
        training=False,
        cache=None,
        decoder_padding=None,
        step=None,
        position_encoding=True,
        step_encoding=True,
        **kwargs
    ):

        with tf.name_scope("UT_decoder"):
            if step_encoding:
                assert step is not None, "step is required when setting step_encoding = True"
            if decoder_padding is not None:
                input_padding = 1 - tf.expand_dims(decoder_padding, -1)
            else:
                input_padding = None
            inputs = input_preprocessing(inputs, input_padding)
            inputs = ut_input_preprocess(
                inputs,
                step,
                position_index=dec_position,
                position_encoding=position_encoding,
                step_encoding=step_encoding,
                **kwargs
            )
            # if not self.preNorm:
            #     inputs = self.input_preNorm(inputs)
            # if training:
            #     inputs = tf.nn.dropout(inputs, rate=self.dropout)
            inputs = super(UniversalTransformerDecoderBLOCK, self).call(
                inputs,
                enc,
                decoder_self_attention_bias,
                attention_bias,
                training=training,
                cache=cache,
                decoder_padding=decoder_padding,
                index=step,
                scale=self.num_units ** -0.5,
                **kwargs
            )
            return inputs
