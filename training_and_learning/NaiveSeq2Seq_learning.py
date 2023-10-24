# -*- coding: utf-8 -*-
# code warrior: Barid
"""
    This is a naive seq2seq model wrapping commone functions.
"""
import sys
import tensorflow as tf
from UNIVERSAL.block import BeamSearchBlock
from UNIVERSAL.basic_metric import seq2seq_metric, mean_metric, WEcos_metric
import numpy as np


class NaiveSeq2Seq(tf.keras.Model):
    def __init__(self, param):
        """
        support loss_type:
            #one_hot
            #sparse
        """
        super(NaiveSeq2Seq, self).__init__(name=param["name"])
        self.beam_search = BeamSearchBlock.BeamSearch()
        self.param = param
        # def build(self, input_shape):
        if 5 in self.param["app"] and len(self.param["app"]) == 5:
            pass
        else:
            self.perplexity = mean_metric.Mean_MetricLayer("perplexity")
            self.total_loss = mean_metric.Mean_MetricLayer("total_loss")
            # self.vae_loss = mean_metric.Mean_MetricLayer("vae_loss")
            self.grad_norm_ratio = mean_metric.Mean_MetricLayer("grad_norm_ratio")
            self.tokenPerS = mean_metric.Mean_MetricLayer("tokens/batch")
            self.finetune = False
            self.seq2seq_loss_FN = seq2seq_metric.CrossEntropy_layer(
                self.param["vocabulary_size"], self.param["label_smoothing"], name="loss",
            )
            self.seq2seq_metric = seq2seq_metric.MetricLayer(self.param["EOS_ID"], prefix="seq2seq")
            # self.WEcos = WEcos_metric.WEcos_MetricLayer()
            if "TFB_freq" in param:
                self.TFB_freq = self.param["TFB_freq"]
            else:
                self.TFB_freq = 100
        ###########gradient tower

    def compile(self, optimizer):
        super(NaiveSeq2Seq, self).compile(optimizer=optimizer)
        self.test_on_batch(np.array([[self.param["EOS_ID"]]]))
        self.summary(print_fn=tf.print)
        self._jit_compile = False

    # training entry
    def train_step(self, data,**kwargs):
        """
            The pipelien:
                1. data pre-processing: ((x, y),) = data
                2. call training_fn: self.seq2seq_training
                3. write tf board:         for m in self.metrics:
                                                tf.summary.scalar(m.name, m.result(), step=self.TFB_freq)

                4. return {m.name: m.result() for m in self.metrics}
        """
        ((x, y),) = data
        self.seq2seq_training(self.call, x, y, self.param["SOS_ID"], training=True, **kwargs)
        return {m.name: m.result() for m in self.metrics}

    def seq2seq_training(self, call_fn, x, y, sos=None, training=True, **kwargs):
        with tf.GradientTape() as model_tape:
            if sos is not None:
                sos_y = tf.pad(y, [[0, 0], [1, 0]], constant_values=sos)[:, :-1]
            else:
                sos_y = y
            re = call_fn((x, sos_y), training=training, **kwargs)
            if len(re)>1:
                x_logits = re[0]
            else:
                re = x_logits
            if "tgt_label" in kwargs:
                y_label = kwargs["tgt_label"]
            else:
                y_label = y
            return self.seq2seq_update(x_logits, y_label, model_tape, **kwargs)

    def seq2seq_update(self, x_logits, y_label, model_tape, **kwargs):
        loss = self.seq2seq_loss_FN([y_label, x_logits], auto_loss=False)
        model_gradients = model_tape.gradient(loss,self.trainable_variables)

        if self.param["clip_norm"] > 0:
            model_gradients, grad_norm = tf.clip_by_global_norm(
                model_gradients, self.param["clip_norm"]
            )
        else:
            grad_norm = tf.linalg.global_norm(model_gradients)
        self.optimizer.apply_gradients(zip(model_gradients, self.trainable_variables))
        self.grad_norm_ratio(grad_norm)
        self.perplexity(tf.math.exp(tf.cast(loss, tf.float32)))
        # if "tgt_label" in kwargs:
        #     y = kwargs["tgt_label"]
        if "tgt_metric" in kwargs:
            y_metric = kwargs["tgt_metric"]
        else:
            y_metric = y_label
        if "src_metric" in kwargs:
            src_metric = kwargs["src_metric"]
        else:
            src_metric = x_logits
        self.seq2seq_metric([y_metric, src_metric])
        # self.WEcos(self.embedding_softmax_layer)
        batch_size = tf.shape(x_logits)[0]
        self.tokenPerS(tf.cast(tf.math.multiply(batch_size, (tf.shape(x_logits)[1])), tf.float32))
        return

    def predict(
        self, autoregressive_fn,cache=None, beam_size=0, max_decode_length=99
    ):
        """Return predicted sequence."""
        decoded_ids, scores = self.beam_search.predict(
            autoregressive_fn,
            self.param["vocabulary_size"],
            eos_id=self.param["EOS_ID"],
            cache=cache,
            max_decode_length=max_decode_length,
            beam_size=beam_size,
        )
        return decoded_ids, scores

    def get_config(self):
        c = self.param
        return c
