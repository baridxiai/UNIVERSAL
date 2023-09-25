# -*- coding: utf-8 -*-
# code warrior: Barid
import tensorflow as tf
from UNIVERSAL.MLM import BERT
from UNIVERSAL.model import MLM_base


def XLM_masking(
    input_ids,
    vocabulary_size,
    all_special_ids=[0],
    masking_id=4,
    mlm_probability=0.15,
    mlm_ratio=[0.8, 0.1, 0.1],
    label_nonmasking=0,
):
    return BERT.BERT_masking(
        input_ids=input_ids,
        vocabulary_size=vocabulary_size,
        all_special_ids=all_special_ids,
        masking_id=masking_id,
        mlm_probability=mlm_probability,
        mlm_ratio=mlm_ratio,
        label_nonmasking=label_nonmasking,
    )


class XLM(MLM_base.MLM_base):
    def __init__(self, param, **kwargs):
        super().__init__(param, **kwargs)
    def seq2seq_update(self, x_logits, y_label, model_tape, **kwargs):
        loss = self.seq2seq_loss_FN([y_label, x_logits], auto_loss=False,detail_x=False)
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
    def pre_training(self, data):
        ((input_src, output_tgt, span,tgt_label,lang_ids),) = data
        src_lang_ids =  tgt_lang_ids = lang_ids
        metric = tgt_label
        if 1 in self.param["app"] and 2 in self.param["app"] and len(self.param["app"]) == 2:
            metric = tf.concat([metric, metric], 0)
            tgt_label = tf.concat([tgt_label, tgt_label], 0)
        _ = self.seq2seq_training(
            self.call,
            input_src,
            output_tgt,
            sos=self.param["EOS_ID"],
            src_id=src_lang_ids,
            tgt_id=tgt_lang_ids,
            tgt_label=tgt_label,
            tgt_metric=metric,
            span=span,
        )
