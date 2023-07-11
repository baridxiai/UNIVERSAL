# -*- coding: utf-8 -*-
# code warrior: Barid

# from UNIVERSAL.basic_optimizer import learning_rate_op, optimizer_op
import tensorflow as tf

from UNIVERSAL.block import UniversalTransformerBlock

from UNIVERSAL.model import ut
from UNIVERSAL.utils import padding_util, cka, staticEmbedding_util
from UNIVERSAL.basic_layer import embedding_layer, layerNormalization_layer
from UNIVERSAL.block import ACT
import json


def input_preprocess(inputs, position_index=None, **kwargs):
    # obseving the model could not understand the distinguish Between
    # step position and position encoding becasu they have the same value.
    if "max_seq" in kwargs:
        max_seq = kwargs["max_seq"]
    else:
        max_seq = 1000
    if position_index is not None:
        length = max_seq
    else:
        length = None
    inputs = staticEmbedding_util.add_position_timing_signal(inputs, 0, position=position_index, length=length)
    return inputs


class ACTencoder(tf.keras.layers.Layer):
    def __init__(self, param, **kwargs):
        super(ACTencoder, self).__init__()
        self.param = param
        self.ACT_encoder = ACT.ACT(
            UniversalTransformerBlock.UniversalTransformerEncoderBLOCK(
                param["num_units"],
                param["num_heads"],
                param["dropout"],
                norm_dropout=param["norm_dropout"],
                preNorm=param["preNorm"],
                epsilon=param["epsilon"],
            ),
            param["dropout"],
        )
        ####### for dynamical controlling steps in inferring###
        self.dynamic_enc = param["num_encoder_steps"]

        # reimplement output layer
        if param["preNorm"]:
            self.final_enc_norm = layerNormalization_layer.LayerNorm(
                epsilon=param["epsilon"], name="encoder_output_norm"
            )

    def call(self, inputs, attention_bias=0, training=False, encoder_padding=None, enc_position=None, vis=False):
        # src = self.LT_encoder.output_norm(self.embedding_softmax_layer(inputs))
        src = inputs
        pre = src
        if training:
            src = tf.nn.dropout(src, rate=self.param["dropout"])
        if vis:
            orgData = tf.zeros([tf.shape(src)[0], tf.shape(src)[1], 0, tf.shape(src)[2]])
            temp = tf.zeros([tf.shape(src)[1], 0])
            sentence = tf.zeros([0])
            halting = tf.zeros([tf.shape(src)[1], 0])
        with tf.name_scope("LT_encoding"):
            mask = tf.zeros([tf.shape(src)[0], tf.shape(src)[1], 1])
            step_inner = pre
            halting_p = 0.
            reminder = 0.
            n_update = 0.
            for step in range(self.dynamic_enc):
                src, halting_p, reminder, n_update = self.ACT_encoder(
                    (src,halting_p, reminder, n_update),
                    attention_bias=attention_bias,
                    step_inner=step_inner,
                    training=training,
                    encoder_padding=encoder_padding,
                    step=step,
                    max_step=self.dynamic_enc,
                    max_seq=self.param["max_sequence_length"],
                    step_encoding=self.param["step_encoding"],
                    position_encoding=self.param["position_encoding"],
                )
                step += 1
                if vis:
                    temp = tf.concat([tf.reduce_mean(cka.feature_space_linear_cka(pre, src), 0), temp], -1)
                    sentence = tf.concat(
                        [tf.reduce_mean(cka.feature_space_linear_cka(pre, src, True), 0), sentence], -1
                    )
                    halting = tf.concat([tf.reduce_mean(halting_p, 0), halting], -1)
                pre = src

        if vis:
            with open("./enc_cka_similarity.json", "w") as outfile:
                json.dump(temp.numpy().tolist(), outfile)
            with open("./enc_cka_similarity_sentence.json", "w") as outfile:
                json.dump(sentence.numpy().tolist(), outfile)
            with open("./enc_halting_pro.json", "w") as outfile:
                json.dump(halting.numpy().tolist(), outfile)
        if self.param["preNorm"]:
            return self.final_enc_norm(src)
        else:
            return src


class ACTdecoder(tf.keras.layers.Layer):
    def __init__(self, param, **kwargs):
        super(ACTdecoder, self).__init__()
        self.param = param
        self.ACT_decoder = ACT.ACT(
            UniversalTransformerBlock.UniversalTransformerDecoderBLOCK(
                param["num_units"],
                param["num_heads"],
                param["dropout"],
                norm_dropout=param["norm_dropout"],
                preNorm=param["preNorm"],
                epsilon=param["epsilon"],
            ),
            param["dropout"],
        )
        ####### for dynamical controlling steps in inferring###
        self.dynamic_dec = param["num_decoder_steps"]

        if param["preNorm"]:
            self.final_dec_norm = layerNormalization_layer.LayerNorm(
                epsilon=param["epsilon"], name="decoder_output_norm"
            )

    def call(
        self,
        inputs,
        enc,
        decoder_self_attention_bias,
        attention_bias,
        training=False,
        cache=None,
        decoder_padding=None,
        dec_position=None,
        vis=False,
    ):
        # tgt = self.LT_decoder.output_norm(self.embedding_softmax_layer(inputs))
        tgt = inputs
        pre = tgt
        if training:
            tgt = tf.nn.dropout(tgt, rate=self.param["dropout"])
        if vis:
            orgData = tf.zeros([tf.shape(tgt)[0], tf.shape(tgt)[1], 0, tf.shape(tgt)[2]])
            temp = tf.zeros([tf.shape(tgt)[1], 0])
            halting = tf.zeros([tf.shape(tgt)[1], 0])
            sentence = tf.zeros([0])
        with tf.name_scope("LT_decoding"):
            mask = tf.zeros([tf.shape(tgt)[0], tf.shape(tgt)[1], 1])
            step_inner = pre
            halting_p = 0.
            reminder = 0.
            n_update = 0.
            for step in range(self.dynamic_dec):
                layer_name = "layer_%d" % step
                tgt, halting_p, reminder, n_update = self.ACT_decoder(
                    (tgt,halting_p, reminder, n_update),
                    enc,
                    decoder_self_attention_bias,
                    attention_bias,
                    training=training,
                    cache=cache[layer_name] if cache is not None else None,
                    decoder_padding=decoder_padding,
                    step_inner=step_inner,
                    step=step,
                    dec_position=dec_position,
                    max_step=self.dynamic_dec,
                    max_seq=self.param["max_sequence_length"],
                    step_encoding=self.param["step_encoding"],
                    position_encoding=self.param["position_encoding"],
                )

                step += 1
                if vis:
                    temp = tf.concat([tf.reduce_mean(cka.feature_space_linear_cka(pre, tgt), 0), temp], -1)
                    sentence = tf.concat(
                        [tf.reduce_mean(cka.feature_space_linear_cka(pre, tgt, True), 0), sentence], -1
                    )
                    halting = tf.concat([tf.reduce_mean(halting_p, 0), halting], -1)
                pre = tgt
        if vis:
            with open("./dec_cka_similarity.json", "w") as outfile:
                json.dump(temp.numpy().tolist(), outfile)
            with open("./dec_cka_similarity_sentence.json", "w") as outfile:
                json.dump(sentence.numpy().tolist(), outfile)
            # orgData = tf.squeeze(cka.feature_space_linear_cka_3d_self(orgData))
            # with open("./dec_orgData.json", "w") as outfile:
            #     json.dump(orgData.numpy().tolist(), outfile)

            with open("./dec_halting_pro.json", "w") as outfile:
                json.dump(halting.numpy().tolist(), outfile)
        if self.param["preNorm"]:
            return self.final_dec_norm(tgt)
        return tgt


class ACT_UT(ut.UniversalTransformer):
    def __init__(self, param, **kwargs):
        super(ut.UniversalTransformer, self).__init__(param)
        ####### for dynamical controlling steps in inferring###
        self.dynamic_enc = param["num_encoder_steps"]
        self.dynamic_dec = param["num_decoder_steps"]

        self.dynamic_halting = 1.0

        self.ACT_encoder = ACTencoder(param)
        self.ACT_decoder = ACTdecoder(param)

        ################### re-write super's encoder and decoder
        self.encoder = self.ACT_encoder
        self.decoder = self.ACT_decoder

    def encoding(self, inputs, attention_bias=0, training=False, encoder_padding=None, enc_position=None, vis=False):
        src = self.embedding_softmax_layer(inputs)
        return self.ACT_encoder(
            src,
            attention_bias=attention_bias,
            training=training,
            encoder_padding=encoder_padding,
            enc_position=enc_position,
            vis=vis,
        )

    def decoding(
        self,
        inputs,
        enc,
        decoder_self_attention_bias,
        attention_bias,
        training=False,
        cache=None,
        decoder_padding=None,
        dec_position=None,
        vis=False,
    ):
        # tgt = self.LT_decoder.output_norm(self.embedding_softmax_layer(inputs))
        tgt = self.embedding_softmax_layer(inputs)
        return self.ACT_decoder(
            tgt,
            enc,
            decoder_self_attention_bias,
            attention_bias,
            training=training,
            cache=cache,
            decoder_padding=decoder_padding,
            dec_position=dec_position,
            vis=vis,
        )
