# coding: utf-8 -*-
# code warrior: Barid
import tensorflow as tf
from UNIVERSAL.utils import cka

# coding: utf-8 -*-
# code warrior: Barid
import tensorflow as tf
class ACT(tf.keras.layers.Layer):
    def __init__(self, layer, dropout):
        """
    Args:
        lyaer:
        dropout: dropout rate inside transition for training.
    """
        super(ACT, self).__init__()
        # layer is the UT block. So pass the "call" function here.
        # e.g., self.layer = Transformer.call
        self.layer = layer
        self.dropout = dropout
        self.act = tf.keras.layers.Dense(1,activation=tf.nn.sigmoid,use_bias=True)

    def build(self, input_shape):
        """Builds the layer."""
        self.num_units = input_shape[-1]
        super(ACT, self).build(input_shape)

    def call(self, act_x, *args, **kwargs):
        if "training" in kwargs:
            training = kwargs["training"]
        else:
            training = False
        with tf.name_scope("Lazy"):
            x,halting_p,reminder,  n_update = act_x
            p = self.act(x)
            still_runing = tf.cast(tf.less(halting_p,1.0), tf.float32)
            new_halted = tf.cast(tf.greater(halting_p+ p * still_runing, 1.0), tf.float32) * still_runing
            still_runing =  tf.cast(tf.less_equal(halting_p+ p * still_runing, 1.0), tf.float32) * still_runing
            halting_p += p * still_runing
            reminder += new_halted * (1.0 - halting_p)
            halting_p += new_halted * reminder
            n_update += still_runing + new_halted
            update_weights = p * still_runing + new_halted * reminder
            x_step = self.layer(x, *args, **kwargs)
            # could be used for deep model, etc..
            # if training:
                # x_step = tf.nn.dropout(x_step, self.dropout)
                # x = tf.nn.dropout(x, self.dropout)
            y = (1-update_weights)*x + update_weights * x_step
            return y, halting_p, reminder, n_update
