# -*- coding: utf-8 -*-
# code warrior: Barid
import tensorflow as tf
import functools
import sys


class Gradient_Tower(tf.keras.optimizers.Optimizer):
    def __init__(self, optimizer, var, tower=1, *args, **kwargs):
        # self.n_gradients = tf.constant(n_gradients, dtype=tf.int32)
        # self.trainable_variables = trainable_variables
        # self.n_gradients = tf.constant(n_gradients, dtype=tf.int32)
        # self.n_acum_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        super(Gradient_Tower, self).__init__(str(optimizer._name) + "_Gradient_Tower")
        self.optimizer = optimizer
        self._use_locking = False
        self.tower = 3
        self._set_hyper("counter", 0)
        self._set_hyper(
            "grad_acc",
            [
                var[i].assign(tf.zeros_like(var[i], dtype=var[i].dtype), use_locking=self._use_locking,)
                for i in range(len(var))
            ],
        )

    def update_model(self, grad, var, *args, **kwargs):
        """Apply conditionally if counter is zero."""
        # grad_acc = self.grad_acc
        tf.print("tower",output_stream=sys.stderr)
        with tf.name_scope(self.optimizer._name):
            if self.tower > 1:

                counter = tf.identity(self._get_hyper("counter"), tf.int16).assign_add(1, use_locking=self._use_locking)
                grad_acc = tf.identity(self._get_hyper("grad_acc"))

                def apply_adam():
                    total_grad = [
                        tf.math.divide(
                            grad_acc[i].assign_add(grad[i], use_locking=self._use_locking),
                            tf.cast(self.tower, grad[i].dtype),
                        )
                        for i in range(len(self.grad_acc))
                    ]
                    adam_op = self.optimizer.apply_gradients(zip(total_grad, var))
                    with tf.control_dependencies([adam_op]):
                        grad_acc_to_zero_op = [
                            self.grad_acc[i].assign(
                                tf.zeros_like(self.grad_acc[i], dtype=self.grad_acc[i].dtype),
                                use_locking=self._use_locking,
                            )
                            for i in range(len(self.grad_acc))
                        ]
                        counter_op = self.update_counter.assign(0, use_locking=self._use_locking)
                    tf.print("updating",output_stream=sys.stderr)
                    return tf.group(adam_op, grad_acc_to_zero_op, counter_op)

                def accumulate_gradient(grad_acc, grad):
                    grad_acc = [
                        grad_acc[i].assign_add(grad[i], use_locking=self._use_locking) for i in range(len(grad_acc))
                    ]
                    self._set_hyper("grad_acc", grad_acc)
                    tf.print("accumulating",output_stream=sys.stderr)
                    # Strip return value

                tf.cond(
                    tf.equal(tf.math.mod(counter, self.tower), 0), apply_adam, accumulate_gradient,
                )
            else:
                self.optimizer.apply_gradients(zip(grad, var))
