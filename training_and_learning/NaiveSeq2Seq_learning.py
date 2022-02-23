# -*- coding: utf-8 -*-
# code warrior: Barid
"""
    This is a naive seq2seq model wrapping commone functions.
"""
import sys
import tensorflow as tf
from UNIVERSAL.block import BeamSearchBlock
from UNIVERSAL.basic_metric import seq2seq_metric, mean_metric, sum_metric
import numpy as np


class NaiveSeq2Seq(tf.keras.Model):
    def __init__(self, param):
        """
        support loss_type:
            #one_hot
            #sparse
        """
        super(NaiveSeq2Seq, self).__init__(name=param["name"])
        # try:
        #     self.gradient_tower = kwargs["gradient_tower"]
        # except Exception:
        #     self.gradient_tower = None
        # self.encoder = lambda x: x
        # self.decoder = lambda x: x
        self.beam_search = BeamSearchBlock.BeamSearch()
        self.param = param
        # def build(self, input_shape):
        self.perplexity = mean_metric.Mean_MetricLayer("perplexity")
        self.total_loss = mean_metric.Mean_MetricLayer("total_loss")
        self.vae_loss = mean_metric.Mean_MetricLayer("vae_loss")
        self.grad_norm_ratio = mean_metric.Mean_MetricLayer("grad_norm_ratio")
        self.tokenPerS = mean_metric.Mean_MetricLayer("tokens/batch")
        self.finetune = False
        self.seq2seq_loss_FN = seq2seq_metric.CrossEntropy_layer(
            self.param["vocabulary_size"], self.param["label_smoothing"], name="loss",
        )
        self.seq2seq_metric = seq2seq_metric.MetricLayer(self.param["EOS_ID"], prefix="seq2seq")
        ###########gradient tower

    def compile(self, optimizer):
        super(NaiveSeq2Seq, self).compile()
        self.test_on_batch(np.array([[self.param["EOS_ID"]]]))
        self.summary(print_fn=tf.print)
        self.optimizer = optimizer
        self._jit_compile = True
        # self._steps_per_execution = tf.constant(steps_per_execution)
        # tf.print([tf.unstack(tf.shape(v)) for v in self.trainable_variables], output_stream=sys.stderr)

    def seq2seq_vae_training(self, call_fn, x, y, eos=None, training=True, annealer=80000, **kwargs):
        with tf.GradientTape() as model_tape:
            if eos is not None:
                eos_y = tf.pad(y, [[0, 0], [1, 0]], constant_values=eos)[:, :-1]
            else:
                eos_y = y
            x_logits, vae_loss = call_fn((x, eos_y), training=training, **kwargs)
            loss = self.seq2seq_loss_FN([y, x_logits], auto_loss=False, pre_sum=False) + (vae_loss) * (
                tf.minimum(1.0, tf.cast(self.optimizer.iterations, tf.float32) / tf.cast(annealer, tf.float32))
            )

            loss = tf.reduce_mean(loss)
        model_gradients = model_tape.gradient(loss, self.trainable_variables)
        if self.param["clip_norm"] > 0:
            model_gradients, grad_norm = tf.clip_by_global_norm(model_gradients, self.param["clip_norm"])
        else:
            grad_norm = tf.linalg.global_norm(model_gradients)
        self.optimizer.apply_gradients(zip(model_gradients, self.trainable_variables))
        self.grad_norm_ratio(grad_norm)
        self.vae_loss(vae_loss)
        self.total_loss(vae_loss + loss)
        self.perplexity(tf.math.exp(loss))
        if "tgt_label" in kwargs:
            y = kwargs["tgt_label"]
        self.seq2seq_metric([y, x_logits])
        batch_size = tf.shape(x)[0]
        self.tokenPerS(tf.cast(tf.math.multiply(batch_size, (tf.shape(x)[1] + tf.shape(y)[1])), tf.float32))
        return loss

    # training entry
    def train_step(self, data):
        """
            return attention_bias, decoder_self_attention_bias, decoder_padding
        """
        ((x, y),) = data
        self.seq2seq_training(self.call, x, y, self.param["SOS_ID"], training=True)

        return {m.name: m.result() for m in self.metrics}

    def seq2seq_training(self, call_fn, x, y, sos=None, training=True, **kwargs):
        with tf.GradientTape() as model_tape:
            if sos is not None:
                sos_y = tf.pad(y, [[0, 0], [1, 0]], constant_values=sos)[:, :-1]
            else:
                sos_y = y
            x_logits = call_fn((x, sos_y), training=training, **kwargs)
            loss = self.seq2seq_loss_FN([y, x_logits], auto_loss=False)
        model_gradients = model_tape.gradient(loss, self.trainable_variables)

        if self.param["clip_norm"] > 0:
            model_gradients, grad_norm = tf.clip_by_global_norm(model_gradients, self.param["clip_norm"])
        else:
            grad_norm = tf.linalg.global_norm(model_gradients)
        self.optimizer.apply_gradients(zip(model_gradients, self.trainable_variables))
        # If n_acum_step reach the n_gradients then we apply accumulated gradients to update the variables otherwise do nothing
        self.grad_norm_ratio(grad_norm)
        # self.total_loss(loss)
        self.perplexity(tf.math.exp(tf.cast(loss, tf.float32)))
        if "tgt_label" in kwargs:
            y = kwargs["tgt_label"]
        self.seq2seq_metric([y, x_logits])
        batch_size = tf.shape(x)[0]
        self.tokenPerS(tf.cast(tf.math.multiply(batch_size, (tf.shape(x)[1] + tf.shape(y)[1])), tf.float32))
        return

    # def predict_step(self,  x_logits, y, sos=None, **kwargs):
    #     """
    #         Reload the default model.predict_step for on-the-fly eval
    #     """
    #     loss = self.seq2seq_loss_FN([y, x_logits], auto_loss=False)
    #     self.perplexity(tf.math.exp(tf.cast(loss, tf.float32)))
    #     if "tgt_label" in kwargs:
    #         y = kwargs["tgt_label"]
    #     self.seq2seq_metric([y, x_logits])
    #     # batch_size = tf.shape(x)[0]
    #     # self.tokenPerS(tf.cast(tf.math.multiply(batch_size, (tf.shape(x)[1] + tf.shape(y)[1])), tf.float32))
    #     return loss
    def predict(self, autoregressive_fn, sos_id=1, eos_id=2, cache=None, beam_size=0, max_decode_length=99):
        """Return predicted sequence."""
        decoded_ids, scores = self.beam_search.predict(
            autoregressive_fn,
            self.param["vocabulary_size"],
            # sos_id=self.param["SOS_ID"],
            eos_id=self.param["EOS_ID"],
            cache=cache,
            max_decode_length=max_decode_length,
            beam_size=self.param["beam_size"],
        )
        return decoded_ids, scores

    def get_config(self):
        c = self.param
        return c
