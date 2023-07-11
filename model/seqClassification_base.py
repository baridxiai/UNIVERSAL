from UNIVERSAL.training_and_learning.seqClassification_learning import seqClassification
import tensorflow as tf
from functools import partial
from UNIVERSAL.basic_metric import seq2seq_metric, mean_metric, acc_metric

class seqClassification_base(seqClassification):
    def __init__(self,model, label_size,param,finetuning_langID=2,test_langID=1,name='seqClassification_base'):
        """
        support loss_type:
            #one_hot
            #sparse
        """
        # super(seqClassification_base, self).__init__(label_size, label_smoothing=0)
        super(seqClassification_base, self).__init__(label_size,param)
        # self.vae_loss = mean_metric.Mean_MetricLayer("vae_loss")
        self.classification_layer = tf.keras.layers.Dense(label_size)
        self.model = model
        self.total_loss = mean_metric.Mean_MetricLayer("loss")
        self.finetuning_langID = finetuning_langID
        self.test_langID = test_langID
    def train_step(self, data):
        ((x,y,ids),) = data
        loss= self.seqClassification_training(self.call, x, y,src_id=ids,tgt_id=ids, rep_model=self.model,fine_tuning=self.classification_layer)
        self.total_loss(loss)
        return {m.name: m.result() for m in self.metrics}
    def test_step(self, data):
        # Unpack the data
        ((x,y,ids),)= data
        y_pred = self.call(x, training=False,src_id=ids,tgt_id=ids)
        self.seqClassification_metric_dev.update_state(y,  tf.nn.softmax(y_pred))
        return {m.name: m.result() for m in self.metrics}
    def call(self, inputs, training=False,  **kwargs):
        src = inputs
        if training:
            lm,representation = self.model((src,src),True, src_id=kwargs["src_id"],tgt_id=kwargs["tgt_id"], cls=False)
        else:
            representation = self.model(src,False,src_id=kwargs["src_id"],tgt_id=kwargs["tgt_id"], beam_size=-1)
        first_column = tf.gather(representation, [0], axis=1)
        first_column = tf.squeeze(first_column,[1])
        if training:
            first_column = tf.nn.dropout(first_column,rate=0.1)
            return self.classification_layer(first_column), lm
        return self.classification_layer(first_column)