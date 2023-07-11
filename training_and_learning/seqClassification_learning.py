# -*- coding: utf-8 -*-
# code warrior: Barid
"""
    This is a naive seq2seq model wrapping commone functions.
"""
import tensorflow as tf
from UNIVERSAL.block import BeamSearchBlock
from UNIVERSAL.basic_metric import seq2seq_metric, mean_metric, acc_metric
import numpy as np


class seqClassification(tf.keras.Model):
    def __init__(self, label_size, param, name="seqClassification"):
        """
        support loss_type:
            #one_hot
            #sparse
        """
        super(seqClassification, self).__init__(name=name)
        # self.vae_loss = mean_metric.Mean_MetricLayer("vae_loss")
        self.grad_norm_ratio = mean_metric.Mean_MetricLayer("grad_norm_ratio")
        self.tokenPerS = mean_metric.Mean_MetricLayer("tokens/batch")
        # self.total_loss = mean_metric.Mean_MetricLayer("acc_loss")
        self.acc_loss = mean_metric.Mean_MetricLayer("acc_loss")
        self.lm_loss = mean_metric.Mean_MetricLayer("lm_loss")
        # self.model_loss = seq2seq_metric.CrossEntropy_layer(
        #                 param["vocabulary_size"], param["label_smoothing"], name="model_loss", mask_id=4,
        # )
        # self.seqClassification_FN = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1)
        self.seqClassification_metric = tf.keras.metrics.CategoricalAccuracy(name="training_acc")
        self.seqClassification_metric_dev = tf.keras.metrics.CategoricalAccuracy(name="val_acc")

    # def compile(self, optimizer):
    #     super(seqClassification, self).compile(optimizer=optimizer)
    #     self.test_on_batch(np.array([[1]]),np.array([[1]]))
    #     self.summary(print_fn=tf.print)
    #     self._jit_compile = False

    def seqClassification_training(self, call_fn, x, y, src_id, tgt_id, rep_model, fine_tuning):
        with tf.GradientTape(persistent=True) as model_tape:
            # if model_trainable_variables is not None:
            #     model_tape.watch(model_trainable_variables)
            model_tape.watch(rep_model.trainable_variables)
            model_tape.watch(fine_tuning.trainable_variables)
            x_logits, lm = call_fn(x, training=True, src_id=src_id, tgt_id=tgt_id)
            acc_loss = tf.reduce_mean(
                tf.keras.losses.categorical_crossentropy(y, x_logits, from_logits=True)
            )
            # lm_loss = self.model_loss([x, lm], auto_loss=False)
            self.acc_loss(acc_loss)
            # self.lm_loss(lm_loss)
            ############### v_1 ############
            # model_gradients = model_tape.gradient(acc_loss,rep_model.trainable_variables)
            # fine_tuning_gradients = model_tape.gradient(acc_loss,fine_tuning.trainable_variables)

            # model_gradients, grad_norm_model = tf.clip_by_global_norm(
            #     model_gradients,5
            # )
            # fine_tuning_gradients, grad_norm_fine_tuning = tf.clip_by_global_norm(
            #     fine_tuning_gradients,5
            # )
            # grad_norm = grad_norm_model + grad_norm_fine_tuning
            # self.optimizer.apply_gradients(zip(fine_tuning_gradients,fine_tuning.trainable_variables))
            # rep_model.optimizer.apply_gradients(zip(model_gradients,rep_model.trainable_variables))
            ############### v_2 ############
            model_gradients = model_tape.gradient(acc_loss, self.trainable_variables)
            _, grad_norm = tf.clip_by_global_norm(model_gradients, 5)
            self.optimizer.apply_gradients(zip(model_gradients, self.trainable_variables))
            self.grad_norm_ratio(grad_norm)
            return self.seqClassification_update(x_logits, y, acc_loss)

    def seqClassification_update(self, x_logits, y_label, loss, **kwargs):
        # soft_targets = tf.one_hot(
        #     tf.cast(y_label, tf.int32), depth=3,
        # )
        # loss = self.seqClassification_FN([soft_targets, x_logits], auto_loss=False)
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
        self.seqClassification_metric.update_state(y_metric, tf.nn.softmax(src_metric))
        # self.WEcos(self.embedding_softmax_layer)
        batch_size = tf.shape(x_logits)[0]
        self.tokenPerS(tf.cast(tf.math.multiply(batch_size, (tf.shape(x_logits)[1])), tf.float32))
        return loss
