# -*- coding: utf-8 -*-
# code warrior: Barid

# from UNIVERSAL.basic_optimizer import learning_rate_op, optimizer_op
import tensorflow as tf
from UNIVERSAL.block import UniversalTransformerBlock
from UNIVERSAL.model import transformer, ut
from UNIVERSAL.utils import padding_util, maskAndBias_util, cka
from UNIVERSAL.basic_layer import embedding_layer, layerNormalization_layer, layerConstaint_layer
import json
import sys


def gru_block(gru, inputs, init, padding_mask):
    batch_size, length, _ = tf.unstack(tf.shape(inputs))
    inputs = padding_util.seq_padding_remover(inputs, padding_mask)
    inputs = tf.expand_dims(inputs, 1)
    if init is not None:
        init = padding_util.seq_padding_remover(init, padding_mask)
        inputs = gru(inputs, init)
    else:
        inputs = gru(inputs)

    inputs = padding_util.seq_padding_restore(inputs, padding_mask)
    return inputs


class UT_GRU_encoder(ut.UTencoder):
    def __init__(self, param, **kwargs):
        # super(UniversalTransformer, self).__init__(param)
        # del self.encoder, self.decoder
        super(UT_GRU_encoder, self).__init__(param)
        # super(transformer.Transformer, self).__init__(param)

        self.enc_gru = tf.keras.layers.GRU(
            self.param["num_units"],
            activation="linear",
            unroll=True,
            use_bias=False,
            # kernel_constraint=layerConstaint_layer.WeightOrthogonalization(),
            # recurrent_constraint=layerConstaint_layer.WeightOrthogonalization(),
        )

    def call(self, inputs, attention_bias=None, training=False, encoder_padding=None, enc_position=None, vis=False):
        src = inputs
        pre = src
        if training:
            src = tf.nn.dropout(src, self.param["dropout"])
        if vis:
            orgData = tf.zeros([tf.shape(src)[0], tf.shape(src)[1], tf.shape(src)[2], 0])
            temp = tf.zeros([tf.shape(src)[1], 0])
            sentence = tf.zeros([0])
        gru_init = None
        gru_init = gru_block(self.enc_gru, src, gru_init, encoder_padding)
        with tf.name_scope("UT_GRU_encoding"):
            for step in range(self.dynamic_enc):
                src = self.ut_encoder(
                    src,
                    attention_bias=attention_bias,
                    training=training,
                    step=step,
                    max_step=self.dynamic_enc,
                    max_seq=self.param["max_sequence_length"],
                    step_encoding=self.param["step_encoding"],
                    position_encoding=self.param["position_encoding"],
                    encoder_padding=encoder_padding,
                )
                if training:
                    src = gru_block(
                        self.enc_gru,
                        tf.nn.dropout(src, self.param["dropout"]),
                        tf.nn.dropout(gru_init, self.param["dropout"]),
                        encoder_padding,
                    )
                else:
                    src = gru_block(self.enc_gru, src, gru_init, encoder_padding)
                gru_init = src
                if vis:
                    step += 1
                    temp = tf.concat([tf.reduce_mean(cka.feature_space_linear_cka(pre, src), 0), temp], -1)
                    sentence = tf.concat(
                        [tf.reduce_mean(cka.feature_space_linear_cka(pre, src, True), 0), sentence], -1
                    )
                    orgData = tf.concat([tf.expand_dims(pre, -1), orgData], -1)
                    pre = src
            if vis:
                with open("./enc_cka_similarity.json", "w") as outfile:
                    json.dump(temp.numpy().tolist(), outfile)
                with open("./enc_cka_similarity_sentence.json", "w") as outfile:
                    json.dump(sentence.numpy().tolist(), outfile)
                orgData = tf.reduce_mean(cka.feature_space_linear_cka_3d_self(orgData), 0)
                with open("./enc_orgData.json", "w") as outfile:
                    json.dump(orgData.numpy().tolist(), outfile)
            if self.param["preNorm"]:
                src = self.final_encoding_norm(src)
            return src


class UT_GRU_decoder(ut.UTdecoder):
    def __init__(self, param, **kwargs):
        # super(UniversalTransformer, self).__init__(param)
        # del self.encoder, self.decoder
        super(UT_GRU_decoder, self).__init__(param)
        # super(transformer.Transformer, self).__init__(param)

        self.dec_gru = tf.keras.layers.GRU(
            self.param["num_units"],
            unroll=True,
            activation="linear",
            use_bias=False,
            # kernel_constraint=layerConstaint_layer.WeightOrthogonalization(),
            # recurrent_constraint=layerConstaint_layer.WeightOrthogonalization(),
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
        tgt = inputs
        pre = tgt
        if training:
            tgt = tf.nn.dropout(tgt, self.param["dropout"])
        gru_init = None
        gru_init = gru_block(self.dec_gru, tgt, gru_init, decoder_padding)
        if vis:
            orgData = tf.zeros([tf.shape(tgt)[0], tf.shape(tgt)[1], tf.shape(tgt)[2], 0])
            temp = tf.zeros([tf.shape(tgt)[1], 0])
            sentence = tf.zeros([0])
        with tf.name_scope("UT_GRU_decoding"):
            for step in range(self.dynamic_dec):
                layer_name = "layer_%d" % step
                tgt = self.ut_decoder(
                    tgt,
                    enc,
                    decoder_self_attention_bias,
                    attention_bias,
                    training=training,
                    cache=cache[layer_name] if cache is not None else None,
                    decoder_padding=decoder_padding,
                    step=step,
                    dec_position=dec_position,
                    max_step=self.dynamic_dec,
                    max_seq=self.param["max_sequence_length"],
                    step_encoding=self.param["step_encoding"],
                    position_encoding=self.param["position_encoding"],
                )
                if training:
                    tgt = gru_block(
                        self.dec_gru,
                        tf.nn.dropout(tgt, self.param["dropout"]),
                        tf.nn.dropout(gru_init, self.param["dropout"]),
                        decoder_padding,
                    )
                else:
                    tgt = gru_block(self.dec_gru, tgt, gru_init, decoder_padding)
                gru_init = tgt
                if vis:
                    step += 1
                    temp = tf.concat([tf.reduce_mean(cka.feature_space_linear_cka(pre, tgt), 0), temp], -1)
                    sentence = tf.concat(
                        [tf.reduce_mean(cka.feature_space_linear_cka(pre, tgt, True), 0), sentence], -1
                    )
                    orgData = tf.concat([tf.expand_dims(pre, -1), orgData], -1)
                    pre = tgt
            if vis:
                with open("./dec_cka_similarity.json", "w") as outfile:
                    json.dump(temp.numpy().tolist(), outfile)
                with open("./dec_cka_similarity_sentence.json", "w") as outfile:
                    json.dump(sentence.numpy().tolist(), outfile)
                orgData = tf.reduce_mean(cka.feature_space_linear_cka_3d_self(orgData), 0)
                with open("./dec_orgData.json", "w") as outfile:
                    json.dump(orgData.numpy().tolist(), outfile)
            if self.param["preNorm"]:
                tgt = self.final_decoding_norm(tgt)
            return tgt


class UniversalTransformer_GRU(transformer.Transformer):
    def __init__(self, param, **kwargs):
        # super(UniversalTransformer, self).__init__(param)
        # del self.encoder, self.decoder
        super(UniversalTransformer_GRU, self).__init__(param)
        # super(transformer.Transformer, self).__init__(param)
        self.param = param
        # setting NaiveSeq2Seq_model.##
        self.ut_GRU_encoder = UT_GRU_encoder(param)
        self.ut_GRU_decoder = UT_GRU_decoder(param)
        self.encoder = self.ut_GRU_encoder
        self.decoder = self.ut_GRU_decoder
        self.embedding_softmax_layer = embedding_layer.EmbeddingSharedWeights(
            param["vocabulary_size"],
            param["num_units"],
            pad_id=param["PAD_ID"],
            name="word_embedding",
            affine=param["affine_we"],
            scale_we=param["scale_we"],
        )
        self.probability_generator = self.embedding_softmax_layer._linear
        ####### for dynamical controlling steps in inferring###
        self.dynamic_enc = param["num_encoder_steps"]
        self.dynamic_dec = param["num_decoder_steps"]

        self.dynamic_halting = 1.0

    def encoding(self, inputs, attention_bias=0, training=False, encoder_padding=None, enc_position=None, vis=False):
        src = self.embedding_softmax_layer(inputs)
        return self.ut_GRU_encoder(
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
        tgt = self.embedding_softmax_layer(inputs)
        return self.ut_GRU_decoder(
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

    def get_config(self):
        c = self.param
        return c
