# -*- coding: utf-8 -*-
# code warrior: Barid
import tensorflow as tf
from UNIVERSAL.basic_layer import attention_layer, ffn_layer, layerNormalization_layer
from UNIVERSAL.utils import staticEmbedding_util

SRC_LANG = 1
TGT_LANG = 2


# class TransformerEncoderBLOCK(tf.keras.layers.Layer):
class TransformerEncoderBLOCK(tf.keras.layers.Layer):
    """
        Navie TransformerEncoderBLOCK implementation including:
        1. 1-layer self attention
        2. 1-layer FFN
    """

    def __init__(
        self,
        num_units=512,
        num_heads=8,
        dropout=0.1,
        norm_dropout=0.1,
        preNorm=False,
        epsilon=1e-6,
        name="TransformerEncoderBLOCK",
        ffn_activation="relu",
    ):
        super(TransformerEncoderBLOCK, self).__init__(name=name)
        self.num_units = num_units
        self.num_heads = num_heads
        self.dropout = dropout
        self.norm_dropout = norm_dropout
        self.n = name
        self.preNorm = preNorm
        self.ffn_activation = ffn_activation
        self.epsilon = epsilon
        if self.preNorm:
            self.att_dropout = self.dropout
            self.ffn_dropout = self.dropout
        else:
            self.att_dropout = self.dropout
            self.ffn_dropout = 0

    def build(self, input_shape):
        self_attention = attention_layer.SelfAttention(
            num_heads=self.num_heads, num_units=self.num_units, dropout=self.att_dropout,
        )

        ffn = ffn_layer.Feed_Forward_Network(
            num_units=4 * self.num_units, activation_filter=self.ffn_activation, dropout=self.ffn_dropout
        )
        self.self_att = layerNormalization_layer.NormBlock(
            self_attention, self.norm_dropout, pre_mode=self.preNorm, epsilon=self.epsilon
        )
        self.ffn = layerNormalization_layer.NormBlock(
            ffn, self.norm_dropout, pre_mode=self.preNorm, epsilon=self.epsilon
        )
        # if self.preNorm:
        #     self.final_norm = layerNormalization_layer.LayerNorm()
        super(TransformerEncoderBLOCK, self).build(input_shape)

    def call(self, inputs, attention_bias=0, training=False, index=None, scale=None, **kwargs):
        with tf.name_scope("Transformer_encoder"):
            inputs = self.self_att(inputs, bias=attention_bias, training=training, scale=scale)
            inputs = self.ffn(inputs, training=training, padding_position=attention_bias)
            return inputs

    def get_config(self):
        c = {
            "num_units": self.num_units,
            "num_heads": self.num_heads,
            "num_encoder_layers": self.num_encoder_layers,
            "dropout": self.dropout,
        }
        return c


# class TransformerDecoderBLOCK(tf.keras.layers.Layer):
class TransformerDecoderBLOCK(tf.keras.layers.Layer):
    def __init__(
        self,
        num_units=512,
        num_heads=8,
        dropout=0.1,
        norm_dropout=0.1,
        preNorm=False,
        epsilon=1e-9,
        name="TransformerDecoderBLOCK",
        ffn_activation="relu",
    ):
        super(TransformerDecoderBLOCK, self).__init__(name=name)
        self.num_units = num_units
        self.num_heads = num_heads
        self.dropout = dropout
        self.norm_dropout = norm_dropout
        self.attention_weights = dict()
        self.n = name
        self.ffn_activation = ffn_activation
        self.preNorm = preNorm
        self.epsilon = epsilon
        if self.preNorm:
            self.att_dropout = self.dropout
            self.ffn_dropout = self.dropout
        else:
            self.att_dropout = self.dropout
            self.ffn_dropout = 0

        # def get_attention_weights(self):
        #     return self.attention_weights

    def build(self, input_shape):

        self_attention = attention_layer.SelfAttention(
            num_heads=self.num_heads, num_units=self.num_units, dropout=self.att_dropout,
        )

        attention = attention_layer.Attention(
            num_heads=self.num_heads, num_units=self.num_units, dropout=self.att_dropout,
        )
        ffn = ffn_layer.Feed_Forward_Network(
            num_units=4 * self.num_units, activation_filter=self.ffn_activation, dropout=self.ffn_dropout
        )
        self.self_att = layerNormalization_layer.NormBlock(
            self_attention, self.norm_dropout, pre_mode=self.preNorm, epsilon=self.epsilon
        )
        self.att = layerNormalization_layer.NormBlock(
            attention, self.norm_dropout, pre_mode=self.preNorm, epsilon=self.epsilon
        )
        self.ffn = layerNormalization_layer.NormBlock(
            ffn, self.norm_dropout, pre_mode=self.preNorm, epsilon=self.epsilon
        )
        # if self.preNorm:
        #     self.final_norm = layerNormalization_layer.LayerNorm()
        super(TransformerDecoderBLOCK, self).build(input_shape)

    def call(
        self,
        inputs,
        enc,
        decoder_self_attention_bias,
        attention_bias,
        training=False,
        cache=None,
        decoder_padding=None,
        index=None,
        scale=None,
        **kwargs
    ):
        with tf.name_scope("Transformer_decoder"):
            inputs = self.self_att(
                inputs, bias=decoder_self_attention_bias, training=training, cache=cache, scale=scale
            )
            inputs = self.att(inputs, y=enc, bias=attention_bias, training=training, scale=scale)
            inputs = self.ffn(inputs, training=training, padding_position=decoder_padding)
            return inputs

    def get_config(self):
        c = {
            "num_units": self.num_units,
            "num_heads": self.num_heads,
            "num_encoder_layers": self.num_encoder_layers,
            "dropout": self.dropout,
        }
        return c
