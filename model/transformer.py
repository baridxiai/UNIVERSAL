# -*- coding: utf-8 -*-
# code warrior: Barid

import tensorflow as tf
from UNIVERSAL.basic_layer import embedding_layer, layerNormalization_layer
from UNIVERSAL.block import TransformerBlock
from UNIVERSAL.utils import padding_util, maskAndBias_util, staticEmbedding_util, cka
from UNIVERSAL.training_and_learning.NaiveSeq2Seq_learning import NaiveSeq2Seq
import json
import sys


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


class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, param, **kwargs):

        super(TransformerEncoder, self).__init__()
        self.param = param

        if param["preNorm"]:
            self.final_encoding_norm = layerNormalization_layer.LayerNorm(epsilon=param["epsilon"])
        if "num_encoder_layers" in self.param:
            self.encoder = []
            for i in range(self.param["num_encoder_layers"]):
                self.encoder.append(
                    TransformerBlock.TransformerEncoderBLOCK(
                        param["num_units"],
                        param["num_heads"],
                        param["dropout"],
                        preNorm=param["preNorm"],
                        epsilon=param["epsilon"],
                        ffn_activation=param["ffn_activation"],
                        name="Encoder__%d" % i,
                    )
                )
        self.dynamic_enc = param["num_encoder_layers"] if "num_encoder_layers" in param else param["num_encoder_steps"]

    def call(self, inputs, attention_bias=0, training=False, encoder_padding=None, enc_position=None, vis=False):
        src = inputs
        if vis:
            orgData = tf.zeros([tf.shape(src)[0], tf.shape(src)[1], tf.shape(src)[2], 0])
            temp = tf.zeros([tf.shape(src)[1], 0])
            sentence = tf.zeros([0])
            pre = src
        with tf.name_scope("encoding"):
            if training:
                src = tf.nn.dropout(src, rate=self.param["dropout"])
            src = input_preprocess(src, enc_position)
            for index in range(self.dynamic_enc):
                src = self.encoder[index](
                    src, attention_bias=attention_bias, encoder_padding=encoder_padding, training=training, index=index,
                )
                if vis:
                    temp = tf.concat([tf.squeeze(cka.feature_space_linear_cka(pre, src), 0), temp], -1)
                    sentence = tf.concat([tf.squeeze(cka.feature_space_linear_cka(pre, src, True), 0), sentence], -1)
                    orgData = tf.concat([tf.expand_dims(pre, -1), orgData], -1)
                    pre = src
        if vis:
            with open("./enc_cka_similarity.json", "w") as outfile:
                json.dump(temp.numpy().tolist(), outfile)
            with open("./enc_cka_similarity_sentence.json", "w") as outfile:
                json.dump(sentence.numpy().tolist(), outfile)
            orgData = tf.squeeze(cka.feature_space_linear_cka_3d_self(orgData), 0)
            with open("./enc_orgData.json", "w") as outfile:
                json.dump(orgData.numpy().tolist(), outfile)
        if self.param["preNorm"]:
            src = self.final_encoding_norm(src)
        return src


class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(self, param, **kwargs):
        super(TransformerDecoder, self).__init__()
        self.param = param
        if param["preNorm"]:
            self.final_decoding_norm = layerNormalization_layer.LayerNorm(epsilon=param["epsilon"])

        if "num_decoder_layers" in self.param:
            self.decoder = []
            for i in range(self.param["num_decoder_layers"]):
                self.decoder.append(
                    TransformerBlock.TransformerDecoderBLOCK(
                        param["num_units"],
                        param["num_heads"],
                        param["dropout"],
                        preNorm=param["preNorm"],
                        epsilon=param["epsilon"],
                        ffn_activation=param["ffn_activation"],
                        name="Decoder__%d" % i,
                    )
                )
        self.dynamic_dec = param["num_decoder_layers"] if "num_decoder_layers" in param else param["num_decoder_steps"]

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
        if vis:
            orgData = tf.zeros([tf.shape(tgt)[0], tf.shape(tgt)[1], tf.shape(tgt)[2], 0])
            temp = tf.zeros([tf.shape(tgt)[1], 0])
            sentence = tf.zeros([0])
            pre = tgt
        with tf.name_scope("decoding"):
            if training:
                tgt = tf.nn.dropout(tgt, rate=self.param["dropout"])
            tgt = input_preprocess(tgt, dec_position, max_seq=self.param["max_sequence_length"])
            for index in range(self.dynamic_dec):
                layer_name = "layer_%d" % index
                tgt = self.decoder[index](
                    tgt,
                    enc,
                    decoder_self_attention_bias,
                    attention_bias,
                    training=training,
                    cache=cache[layer_name] if cache is not None else None,
                    decoder_padding=decoder_padding,
                    index=index,
                )
                if vis:
                    temp = tf.concat([tf.squeeze(cka.feature_space_linear_cka(pre, tgt), 0), temp], -1)
                    sentence = tf.concat([tf.squeeze(cka.feature_space_linear_cka(pre, tgt, True), 0), sentence], -1)
                    orgData = tf.concat([tf.expand_dims(pre, -1), orgData], -1)
                    pre = tgt
        if vis:
            with open("./dec_cka_similarity.json", "w") as outfile:
                json.dump(temp.numpy().tolist(), outfile)
            with open("./dec_cka_similarity_sentence.json", "w") as outfile:
                json.dump(sentence.numpy().tolist(), outfile)
            orgData = tf.squeeze(cka.feature_space_linear_cka_3d_self(orgData))
            with open("./dec_orgData.json", "w") as outfile:
                json.dump(orgData.numpy().tolist(), outfile)
        if self.param["preNorm"]:
            tgt = self.final_decoding_norm(tgt)
        return tgt


class Transformer(NaiveSeq2Seq):
    def __init__(self, param, **kwargs):

        super(Transformer, self).__init__(param)
        self.param = param
        ##
        self.embedding_softmax_layer = embedding_layer.EmbeddingSharedWeights(
            param["vocabulary_size"],
            param["num_units"],
            pad_id=param["PAD_ID"],
            name="word_embedding",
            affine=param["affine_we"],
            scale_we=param["scale_we"],
        )
        self.probability_generator = self.embedding_softmax_layer._linear
        self.encoder = TransformerEncoder(param)
        self.decoder = TransformerDecoder(param)
        ## setting NaiveSeq2Seq_model.##

    def get_config(self):
        c = self.param
        return c

    def encoding(self, inputs, attention_bias=0, training=False, encoder_padding=None, enc_position=None, vis=False):
        src = self.embedding_softmax_layer(inputs)
        src = self.encoder(
            src,
            attention_bias=attention_bias,
            training=training,
            encoder_padding=encoder_padding,
            enc_position=enc_position,
            vis=vis,
        )
        return src

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
        tgt = self.decoder(
            inputs,
            enc,
            decoder_self_attention_bias,
            attention_bias,
            training=training,
            cache=cache,
            decoder_padding=decoder_padding,
            dec_position=dec_position,
            vis=vis,
        )
        return tgt

    def pre_processing(self, src, tgt):
        attention_bias = padding_util.get_padding_bias(src)
        encoder_padding = padding_util.get_padding(src)
        decoder_padding = padding_util.get_padding(tgt)
        decoder_self_attention_bias = maskAndBias_util.get_decoder_self_attention_bias(tf.shape(tgt)[1])
        return attention_bias, decoder_self_attention_bias, encoder_padding, decoder_padding

    def prepare_cache(self, src, src_id, sos_id):
        batch_size = tf.shape(src)[0]
        initial_ids = tf.zeros([batch_size], dtype=tf.int32) + self.param["SOS_ID"]
        attention_bias = padding_util.get_padding_bias(src)
        encoder_padding = padding_util.get_padding(src)
        enc = self.encoding(src, attention_bias=attention_bias, training=False, encoder_padding=encoder_padding)

        init_decode_length = 0
        dim_per_head = self.param["num_units"] // self.param["num_heads"]
        cache = dict()
        cache = {
            "layer_%d"
            % layer: {
                "k": tf.zeros([batch_size, self.param["num_heads"], init_decode_length, dim_per_head]),
                "v": tf.zeros([batch_size, self.param["num_heads"], init_decode_length, dim_per_head]),
            }
            # for layer in range(self.num_decoder_steps)
            for layer in range(self.decoder.dynamic_dec)
        }
        cache["decoder_padding"] = padding_util.get_padding(tf.ones([batch_size, 1], dtype=tf.int32))
        # cache["decoder_padding"] =  None
        cache["enc"] = enc
        cache["initial_ids"] = initial_ids
        cache["attention_bias"] = attention_bias
        return cache, batch_size

    def autoregressive_fn(self, max_decode_length, lang_embedding=0, tgt_domain_id=0):
        """Returns a decoding function that calculates logits of the next tokens."""
        decoder_self_attention_bias = maskAndBias_util.get_decoder_self_attention_bias(max_decode_length)

        def symbols_to_logits_fn(ids, i, cache):
            decoder_input_id = tf.cast(ids[:, -1:], tf.int32)
            self_attention_bias = decoder_self_attention_bias[:, i : i + 1, : i + 1]
            dec = self.decoding(
                decoder_input_id,
                cache.get("enc"),
                decoder_self_attention_bias=self_attention_bias,
                attention_bias=cache.get("attention_bias"),
                dec_position=i,
                decoder_padding=cache.get("decoder_padding"),
                training=False,
                cache=cache,
            )
            logits = self.probability_generator(dec)
            logits = tf.squeeze(logits, axis=[1])
            return logits, cache

        return symbols_to_logits_fn

    def get_config(self):
        c = {"param": self.param}
        return c

    def forward(
        self,
        src,
        tgt,
        training=True,
        attention_bias=0,
        decoder_self_attention_bias=0,
        cache=None,
        encoder_padding=None,
        decoder_padding=None,
        enc_position=None,
        dec_position=None,
        vis=False,
    ):
        enc = self.encoding(
            src, attention_bias, training=training, encoder_padding=encoder_padding, enc_position=enc_position, vis=vis
        )
        dec = self.decoding(
            tgt,
            enc,
            decoder_self_attention_bias,
            attention_bias,
            training=training,
            cache=cache,
            dec_position=dec_position,
            vis=vis,
            decoder_padding=decoder_padding,
        )
        # logits = self.embedding_softmax_layer(dec, linear=True)
        logits = self.probability_generator(dec)
        return logits

    def call(self, inputs, training=False, **kwargs):
        vis = False
        if "vis" in kwargs:
            vis = kwargs["vis"]
        if training:
            src, tgt = inputs[0], inputs[1]
            attention_bias, decoder_self_attention_bias, encoder_padding, decoder_padding, = self.pre_processing(
                src, tgt
            )
            logits = self.forward(
                src,
                tgt,
                training=training,
                attention_bias=attention_bias,
                decoder_self_attention_bias=decoder_self_attention_bias,
                encoder_padding=encoder_padding,
                decoder_padding=decoder_padding,
                vis=vis,
            )
            return logits
        else:
            org_enc = self.encoder.dynamic_enc
            org_dec = self.decoder.dynamic_dec
            if "enc" in kwargs:
                self.encoder.dynamic_enc = kwargs["enc"]
            if "dec" in kwargs:
                self.decoder.dynamic_dec = kwargs["dec"]
            beam_szie = 4
            if "beam_size" in kwargs:
                beam_szie = kwargs["beam_size"]
            tgt = None
            src = inputs
            _, length = tf.unstack(tf.shape(src))
            cache, batch_size = self.prepare_cache(src, self.param["SOS_ID"], self.param["SOS_ID"])
            max_length = self.param["max_sequence_length"]
            autoregressive_fn = self.autoregressive_fn(max_length,)
            re, score = self.predict(
                autoregressive_fn,
                eos_id=self.param["EOS_ID"],
                max_decode_length=int(length + 50),
                cache=cache,
                beam_size=beam_szie,
            )
            tf.print("enc", self.encoder.dynamic_enc, "dec", self.decoder.dynamic_dec, output_stream=sys.stdout)
            top_decoded_ids = re[:, 0, 1:]
            self.encoder.dynamic_enc = org_enc
            self.decoder.dynamic_dec = org_dec
            del cache, re, score
            return top_decoded_ids
