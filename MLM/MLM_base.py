# -*- coding: utf-8 -*-
# code warrior: Barid
from UNIVERSAL.model import language_model
from UNIVERSAL.basic_layer import embedding_layer
from UNIVERSAL.utils import padding_util
import tensorflow as tf
from functools import partial
from UNIVERSAL.basic_metric import acc_metric
# from UNIVERSAL.block import TransformerBlock
# import CLPM
import sys


class MLM_base(language_model.LM_base):
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

    def pre_training(self, data):
        # (input_src, output_tgt, tgt_label, src_lang_ids, tgt_lang_ids, span) = self.MLM(data)
        ((input_src, output_tgt, span,tgt_label,lang_ids),) = data
        src_lang_ids =  tgt_lang_ids = lang_ids
        _ = self.seq2seq_training(
            self.call,
            input_src,
            output_tgt,
            # sos=self.param["EOS_ID"],
            src_id=src_lang_ids,
            tgt_id=tgt_lang_ids,
            tgt_label=tgt_label,
            tgt_metric=tf.where(tf.equal(input_src, self.param["MASK_ID"]), tgt_label, input_src),
            span=span,
        )

    def train_step(self, data):
        if 4 in self.param["app"]:
            self.unmt_app(data)
        else:
            self.pre_training(data)
        return {m.name: m.result() for m in self.metrics}

    def unmt_app(self, data):
        (synthetic_y, x, synthetic_x, y) = self.on_the_fly_back_translation(data)
        (input_src, output_tgt, tgt_label, src_lang_ids, tgt_lang_ids, span) = self.MLM(data)
        synthetic_x, input_src = padding_util.pad_tensors_to_same_length(synthetic_x, input_src)
        synthetic_y, input_src = padding_util.pad_tensors_to_same_length(synthetic_y, input_src)
        x, output_tgt = padding_util.pad_tensors_to_same_length(x, output_tgt)
        y, output_tgt = padding_util.pad_tensors_to_same_length(y, output_tgt)
        output_tgt = tf.concat([output_tgt, y, x], 0)
        src_lang_ids = tf.concat([src_lang_ids, src_lang_ids,], 0,)
        tgt_lang_ids = tf.concat([tgt_lang_ids, tgt_lang_ids,], 0,)
        x, input_src = padding_util.pad_tensors_to_same_length(x, input_src)
        y, x = padding_util.pad_tensors_to_same_length(y, x)
        tgt_label, y = padding_util.pad_tensors_to_same_length(tgt_label, y)
        tgt_metric = tf.where(tf.equal(input_src, self.param["MASK_ID"]), tgt_label, input_src)
        tgt_label = tgt_metric = tf.concat([tgt_metric, y, x], 0)
        input_src = tf.concat([input_src, synthetic_x, synthetic_y], 0)
        output_tgt, tgt_metric = padding_util.pad_tensors_to_same_length(output_tgt, tgt_metric)
        tgt_label, output_tgt = padding_util.pad_tensors_to_same_length(tgt_label, output_tgt)

    def on_the_fly_back_translation(self, data, src_id=1, tgt_id=2):
        ((x_input_span, _, _, x_label, y_input_span, _, _, y_label),) = data
        x = tf.where(tf.equal(x_input_span, self.param["MASK_ID"]), x_label, x_input_span)
        y = tf.where(tf.equal(y_input_span, self.param["MASK_ID"]), y_label, y_input_span)

        def _greedy_decoding(inputs, src_id=src_id, tgt_id=tgt_id):
            ids = tf.stop_gradient(
                self.call(
                    inputs,
                    training=False,
                    src_id=src_id,
                    tgt_id=tgt_id,
                    sos_id=self.param["EOS_ID"],
                    beam_size=1,
                )
            )
            return ids

        synthetic_y = _greedy_decoding(x, src_id=1, tgt_id=2)
        synthetic_x = _greedy_decoding(y, src_id=2, tgt_id=1)
        return synthetic_y, x, synthetic_x, y
