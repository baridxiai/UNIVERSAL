# -*- coding: utf-8 -*-
# code warrior: Barid

# from UNIVERSAL.basic_optimizer import learning_rate_op, optimizer_op
import tensorflow as tf
from UNIVERSAL.block import UniversalTransformerBlock
from UNIVERSAL.model import transformer
from UNIVERSAL.utils import padding_util, maskAndBias_util, cka
from UNIVERSAL.basic_layer import embedding_layer, layerNormalization_layer
import json
import sys


class UniversalTransformer(transformer.Transformer):
    def __init__(self, param, **kwargs):
        # super(UniversalTransformer, self).__init__(param)
        # del self.encoder, self.decoder
        super(UniversalTransformer, self).__init__(param)
        # super(transformer.Transformer, self).__init__(param)
        self.param = param
        # setting NaiveSeq2Seq_model.##
        self.ut_encoder = UniversalTransformerBlock.UniversalTransformerEncoderBLOCK(
            num_units=param["num_units"],
            num_heads=param["num_heads"],
            dropout=param["dropout"],
            norm_dropout=param["norm_dropout"],
            preNorm=param["preNorm"],
            epsilon=param["epsilon"],
        )
        self.ut_decoder = UniversalTransformerBlock.UniversalTransformerDecoderBLOCK(
            num_units=param["num_units"],
            num_heads=param["num_heads"],
            dropout=param["dropout"],
            norm_dropout=param["norm_dropout"],
            preNorm=param["preNorm"],
            epsilon=param["epsilon"],
        )

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

        # reimplement output layer
        # self.probability_generator = tf.keras.layers.Dense(param["vocabulary_size"], use_bias=False)

        if param["preNorm"]:
            self.final_encoding_norm = layerNormalization_layer.LayerNorm(
                epsilon=param["epsilon"], name="encoder_output_norm"
            )
            self.final_decoding_norm = layerNormalization_layer.LayerNorm(
                epsilon=param["epsilon"], name="decoder_output_norm"
            )

    def get_config(self):
        c = self.param
        return c

    def encoding(self, inputs, attention_bias=None, training=False, enc_position=None, vis=False):
        src = self.embedding_softmax_layer(inputs)
        pre = src
        if training:
            src = tf.nn.dropout(src, self.param["dropout"])
        if vis:
            orgData = tf.zeros([tf.shape(src)[0], tf.shape(src)[1], tf.shape(src)[2], 0])
            temp = tf.zeros([tf.shape(src)[1], 0])
            sentence = tf.zeros([0])
        with tf.name_scope("UT_encoding"):
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
                )
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
            # return self.encoding_output(src)

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
        pre = tgt
        if training:
            tgt = tf.nn.dropout(tgt, self.param["dropout"])
        if vis:
            orgData = tf.zeros([tf.shape(tgt)[0], tf.shape(tgt)[1], tf.shape(tgt)[2], 0])
            temp = tf.zeros([tf.shape(tgt)[1], 0])
            sentence = tf.zeros([0])
        with tf.name_scope("UT_decoding"):
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
