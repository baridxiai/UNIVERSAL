# -*- coding: utf-8 -*-
# code warrior: Barid
from UNIVERSAL.model import transformer
from UNIVERSAL.basic_layer import embedding_layer
from UNIVERSAL.utils import padding_util
import tensorflow as tf
from functools import partial
from UNIVERSAL.basic_metric import acc_metric
# from UNIVERSAL.block import TransformerBlock
# import CLPM
import sys

class LM_base(transformer.Transformer):
    def __init__(self, param, **kwargs):
        super().__init__(param, **kwargs)
        self.lang_encoding = embedding_layer.EmbeddingSharedWeights(
            20,
            param["num_units"],
            pad_id=param["PAD_ID"],
            affine=param["affine_we"],
            scale_we=param["scale_we"],
            name="lang_enc"+"_"+str(param["scale_we"]),
        )
        self.cls = self.add_weight(
            shape=[param["num_units"]],
            dtype="float32",
            name="cls",
            initializer=tf.random_normal_initializer(mean=0.0, stddev=param["num_units"] ** -0.5),
        )
        self.seq2seq_metric = acc_metric.Word_Accuracy_Metric("token_acc")

    def _run_encoder(
        self,
        src,
        src_id,
        attention_bias,
        training,
        encoder_padding,
        enc_position,
        cls=False,
        vis=False,
    ):
        src = self.embedding_softmax_layer(src)
        if cls:
            cls_token = tf.ones([tf.shape(src)[0],1,1]) * self.cls
            src = tf.concat((cls_token,src),1)
            attention_bias = padding_util.get_embedding_padding(src)
            encoder_padding = 1- attention_bias
            attention_bias = tf.expand_dims(attention_bias, axis=-1)
        if src_id != None:
            src_id = self.lang_encoding(src_id)
            src += src_id
        # self.encoding(src[5:6],attention_bias[5:6],training=True, encoder_padding=encoder_padding[5:6])
        # t = 0
        # t_list = []
        # for i in range(12):
        # # t += tf.reduce_mean(model.encoder.encoder[i].self_att.layer.attention_weights,1).numpy()[0][[1,2,4,6,7,9]]
        #     weights = tf.reduce_mean(
        #         self.encoder.encoder[i].self_att.layer.attention_weights, 1
        #     ).numpy()[0]
        #     t_list.append(weights.tolist())
        #     t += weights
        enc = self.encoding(
            src,
            attention_bias,
            training=training,
            encoder_padding=encoder_padding,
            enc_position=enc_position,
            vis=vis,
        )
        return enc

    def _run_decoder(
        self,
        tgt,
        tgt_id,
        encATT,
        decoder_self_attention_bias,
        attention_bias,
        training,
        cache,
        dec_position,
        decoder_padding,
        vis=False,
    ):
        tgt = self.embedding_softmax_layer(tgt)
        if tgt_id != None:
            tgt_id = self.lang_encoding(tgt_id)
            tgt += tgt_id
        dec = self.decoding(
            tgt,
            encATT,
            decoder_self_attention_bias,
            attention_bias,
            training=training,
            cache=cache,
            dec_position=dec_position,
            vis=vis,
            decoder_padding=decoder_padding,
        )
        return dec
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
        src_id=None,
        tgt_id=None,
        cls=False,
    ):

        run_encoder = partial(
            self._run_encoder,
            attention_bias=attention_bias,
            training=training,
            encoder_padding=encoder_padding,
            enc_position=enc_position,
            cls=cls,
            vis=vis,
        )
        run_decoder = partial(
            self._run_decoder,
            decoder_self_attention_bias=decoder_self_attention_bias,
            attention_bias=attention_bias,
            training=training,
            cache=cache,
            dec_position=dec_position,
            vis=vis,
            decoder_padding=decoder_padding,
        )
        if 1 in self.param["app"] and len(self.param["app"]) == 1:
            logits = run_encoder(src, src_id)
        if 2 in self.param["app"] and len(self.param["app"]) == 1:
            logits = run_decoder(tgt, tgt_id, None)
        if 3 in self.param["app"] and len(self.param["app"]) == 1:
            enc = run_encoder(src, src_id)
            logits = run_decoder(tgt, tgt_id, enc)
        if 1 in self.param["app"] and 2 in self.param["app"] and len(self.param["app"]) == 2:
            enc = run_encoder(src, src_id,)
            dec = run_decoder(tgt, tgt_id, None)
            logits = tf.concat([enc, dec], 0)
        return logits

    def call(self, inputs, training=False, cls=False,**kwargs):
        vis = False
        if "vis" in kwargs:
            vis = kwargs["vis"]
        if training:
            src_id = kwargs["src_id"]
            tgt_id = kwargs["tgt_id"]
            # span = kwargs["span"]
            # tgt_label = kwargs["tgt_label"]
            src, tgt = inputs[0], inputs[1]
            (
                attention_bias,
                decoder_self_attention_bias,
                encoder_padding,
                decoder_padding,
            ) = self.pre_processing(src, tgt)
            logits_raw = self.forward(
                src,
                tgt,
                training=training,
                attention_bias=attention_bias,
                decoder_self_attention_bias=decoder_self_attention_bias,
                encoder_padding=encoder_padding,
                decoder_padding=decoder_padding,
                vis=vis,
                src_id=src_id,
                tgt_id=tgt_id,
                cls=cls
            )
            logits = self.probability_generator(logits_raw)
            return logits, logits_raw
        else:
            # default
            src_id = 1
            tgt_id = 1
            beam_size = 4
            sos_id = self.param["SOS_ID"]
            eos_id = self.param["EOS_ID"]

            if "sos_id" in kwargs:
                sos_id = kwargs["sos_id"]
            if "eos_id" in kwargs:
                eos_id = kwargs["eos_id"]
            if "src_id" in kwargs:
                src_id = kwargs["src_id"]
            if "tgt_id" in kwargs:
                tgt_id = kwargs["tgt_id"]
            if "beam_size" in kwargs:
                beam_size = kwargs["beam_size"]
            tgt = None
            src = inputs
            _, length = tf.unstack(tf.shape(src))
            src_id_token = tf.ones_like(src, dtype=tf.int32) * src_id
            cache, batch_size = self.prepare_cache(src, self.lang_encoding(src_id_token), sos_id)
            if 1 in self.param["app"] and len(self.param["app"]) == 1:
                if self.param["ConRep"]:
                    return cache.get("enc")
                ids = self.probability_generator(cache.get("enc"))
                return tf.argmax(ids, -1)

            # beam search
            max_length = self.param["max_sequence_length"]
            tgt_id_token = (
                tf.expand_dims(tf.ones_like(cache["initial_ids"], dtype=tf.int32), 1) * tgt_id
            )
            autoregressive_fn = self.autoregressive_fn(
                max_length, lang_embedding=self.lang_encoding(tgt_id_token), beam_size=beam_size
            )
            if self.decoder.dynamic_dec != 0:
                re, score = self.predict(
                    autoregressive_fn,
                    eos_id=eos_id,
                    max_decode_length=int(length + 50),
                    cache=cache,
                    beam_size=beam_size,
                )
                tf.print(
                    "enc",
                    self.encoder.dynamic_enc,
                    "dec",
                    self.decoder.dynamic_dec,
                    output_stream=sys.stdout,
                )
            else:
                re = self.probability_generator(cache["enc"])
            top_decoded_ids = re[:, 0, 1:]
            del cache, re, score
            return top_decoded_ids
