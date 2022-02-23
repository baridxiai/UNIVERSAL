# -*- coding: utf-8 -*-
# code warrior: Barid
import tensorflow as tf


class WeightOrthogonalization(tf.keras.constraints.Constraint):
    def __init__(self, beta=0.001, axis=0):
        self.beta = beta
        super(WeightOrthogonalization,
              self).__init__()

    def __call__(self, w):
        orthogonalised_w = (1 + self.beta) * w - self.beta * tf.matmul(
            w, tf.matmul(w, w, transpose_a=True)
        )
        return orthogonalised_w

    def get_config(self):
        return {"beta": self.beta}


class ScaleWeight(tf.keras.constraints.Constraint):
    def __init__(self, scale=0):
        self.c = scale
        super(ScaleWeight, self).__init__(name="ScaleWeight")

    def __call__(self, x):
        x = x * self.c
        return x

    def get_config(self):
        return {"name": self.__class__.__name__, "c": self.c}


class WeightClipper(tf.keras.constraints.Constraint):
    """Clips the weights incident to each hidden unit to be inside a range
    """

    def __init__(self, c=0):
        self.c = c
        super(WeightClipper, self).__init__()

    def __call__(self, p):
        if self.c > 0:
            p = tf.clip_by_value(p, -self.c, self.c)
        return p

    def get_config(self):
        return {"name": self.__class__.__name__, "c": self.c}
