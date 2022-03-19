# -*- coding: utf-8 -*-
# code warrior: Barid
"""
    This is a naive seq2seq model wrapping commone functions.
"""
import tensorflow as tf
from training_and_learning import NaiveVaeSeq2Seq_learning
import numpy as np

class VaeNaiveSeq2Seq(NaiveVaeSeq2Seq_learning):
    def seq2seq_training(self, call_fn, x, y, eos=None, training=True, annealer=80000, **kwargs):
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
