# -*- coding: utf-8 -*-
# code warrior: Barid
import tensorflow as tf
import numpy as np


class LearningRateFn_WL(object):
    def __init__(self, hidden_size=1024, warmup_steps=3000):
        self.hidden_size = hidden_size
        self.warmup_steps = float(warmup_steps)

    def __call__(self, global_step):
        """Calculate learning rate with linear warmup and rsqrt decay."""
        step = float(global_step)
        self.warmup_steps = float(self.warmup_steps)
        # learning_rate = self.learning_rate
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return float(tf.math.rsqrt(tf.cast(self.hidden_size, tf.float32)) * tf.math.minimum(arg1, arg2))
        # learning_rate = self.hidden_size ** -0.5
        # step * (self.warmup_steps ** -1.5)
        # Apply linear warmup
        # learning_rate *= np.minimum(1.0, step / self.warmup_steps)
        # # # Apply rsqrt decay
        # learning_rate /= np.sqrt(np.maximum(step, self.warmup_steps))
        # return learning_rate


class LearningRateSchedule_WARMUP(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate schedule."""

    def __init__(self, initial_learning_rate, hidden_size, warmup_steps, ga_step=1):
        """Initialize configuration of the learning rate schedule.
    Args:
      initial_learning_rate: A float, the initial learning rate.
      hidden_size: An integer, the model dimension in the hidden layers.
      warmup_steps: An integer, the number of steps required for linear warmup.
      ga_step: An integer, the number of steps for accumulating gradients
    """
        super(LearningRateSchedule_WARMUP, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.hidden_size = tf.cast(hidden_size, tf.float32)
        self.warmup_steps = warmup_steps
        self.warmup_steps_tensor = tf.cast(warmup_steps, tf.float32)
        self.ga_step = tf.cast(ga_step, tf.float32)

    def __call__(self, global_step):
        """Calculate learning rate with linear warmup and rsqrt decay.
    Args:
      global_step: An integer, the current global step used for learning rate
        calculation.
    Returns:
      A float, the learning rate needs to be used for current global step.
    """
        with tf.name_scope("learning_rate_schedule"):
            global_step = tf.cast(global_step, tf.float32) / self.ga_step
            learning_rate = self.initial_learning_rate
            learning_rate *= self.hidden_size ** -0.5
            # Apply linear warmup
            learning_rate *= tf.minimum(1.0, global_step / self.warmup_steps_tensor)
            # Apply rsqrt decay
            learning_rate *= tf.math.rsqrt(tf.maximum(global_step, self.warmup_steps_tensor))
            return learning_rate

            # step = tf.cast(global_step, tf.float32)
            # self.warmup_steps = tf.cast(self.warmup_steps, tf.float32)
            # learning_rate = self.learning_rate
            # arg1 = (step) ** -0.5
            # arg2 = step * (self.warmup_steps ** -1.5)
            # return (self.hidden_size ** -0.5) * tf.minimum(arg1, arg2)

    def get_config(self):
        """Get the configuration of the learning rate schedule."""
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "hidden_size": self.hidden_size,
            "warmup_steps": self.warmup_steps,
        }


# class LearningRateScheduler(tf.keras.callbacks.Callback):
#     """Keras callback to schedule learning rate.

#   TODO(tianlin): Refactor this scheduler and LearningRateBatchScheduler in
#   official/resnet/keras/keras_common.py.
#   """

#     def __init__(self, schedule, init_steps=0, verbose=False):
#         super(LearningRateScheduler, self).__init__()
#         self.schedule = schedule
#         self.verbose = verbose
#         if init_steps is None:
#             init_steps = 0.0
#         self.steps = float(init_steps)  # Total steps during training.

#     def on_epoch_begin(self, epoch, logs=None):
#         if not hasattr(self.model.optimizer, "lr"):
#             raise ValueError('Optimizer must have a "lr" attribute.')
#         if not hasattr(self.model.optimizer, "iterations"):
#             raise ValueError('Optimizer must have a "iterations" attribute.')

#     def on_train_batch_begin(self, batch, logs=None):
#         """Adjusts learning rate for each train batch."""
#         if self.verbose > 0:
#             iterations = tf.keras.backend.get_value(self.model.optimizer.iterations)
#             print("Original iteration %d" % iterations)

#         self.steps += 1.0
#         try:  # new API
#             lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
#             lr = self.schedule(self.steps, lr)
#         except TypeError:  # Support for old API for backward compatibility
#             lr = self.schedule(self.steps)
#         if not isinstance(lr, (float, np.float32, np.float64)):
#             raise ValueError('The output of the "schedule" function ' "should be float.")
#         tf.keras.backend.set_value(self.model.optimizer.lr, lr)
#         tf.keras.backend.set_value(self.model.optimizer.iterations, self.steps)

#         if self.verbose > 0:
#             print(
#                 "Batch %05d Step %05d: LearningRateScheduler setting learning "
#                 "rate to %s." % (batch + 1, self.steps, lr)
#             )

#     def on_batch_end(self, batch, logs=None):
#         logs = logs or {}
#         logs["lr"] = tf.keras.backend.get_value(self.model.optimizer.lr)
#         # logs['steps'] = self.steps

#     def on_epoch_end(self, epoch, logs=None):
#         logs = logs or {}
#         logs["lr"] = tf.keras.backend.get_value(self.model.optimizer.lr)
#         logs["steps"] = self.steps


class LearningRateVisualization(tf.keras.callbacks.Callback):
    """Keras callback to schedule learning rate.

  TODO(tianlin): Refactor this scheduler and LearningRateBatchScheduler in
  official/resnet/keras/keras_common.py.
  """

    def __init__(self):
        super(LearningRateVisualization, self).__init__()

    def on_batch_begin(self, batch, logs=None):
        logs = logs or {}
        logs["lr"] = tf.keras.backend.get_value(self.model.optimizer._lr_t)


class warmup_lr(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Applys a warmup schedule on a given learning rate decay schedule."""

    def __init__(
        self,
        initial_learning_rate,
        warmup_steps,
        decay_steps,
        end_learning_rate=0.000001,
        cycle=False,
        power=4.0,
        name=None,
    ):
        # super(WarmUp, self).__init__()
        self.initial_learning_rate = self.current_lr = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.power = power
        self.cycle = cycle
        self.decay_steps = decay_steps
        self.end_learning_rate = end_learning_rate
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "WarmUp"):
            # Implements polynomial warmup. i.e., if global_step < warmup_steps, the
            # learning rate will be `global_step/num_warmup_steps * init_lr`.
            global_step_float = step * 1.0
            warmup_steps_float = self.warmup_steps * 1.0
            warmup_percent_done = global_step_float / warmup_steps_float
            warmup_learning_rate = self.initial_learning_rate * warmup_percent_done ** self.power
            return tf.cond(
                global_step_float < warmup_steps_float,
                lambda: warmup_learning_rate,
                lambda: self.polynomialDecay(step),
                name="dynamic_lr",
            )

    def polynomialDecay(self, step):

        global_step_recomp = tf.minimum(step, self.decay_steps)

        p = global_step_recomp / self.decay_steps
        lr = (self.initial_learning_rate - self.end_learning_rate) * (1 - p) ** self.power + self.end_learning_rate
        return lr

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_schedule_fn": self.decay_schedule_fn,
            "warmup_steps": self.warmup_steps,
            "power": self.power,
            "decay_steps": self.decay_steps,
            "end_learning_rate": self.end_learning_rate,
            "cycle": self.cycle,
            "name": self.name,
        }
