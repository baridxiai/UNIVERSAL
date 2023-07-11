# -*- coding: utf-8 -*-
# code warrior: Barid
import tensorflow as tf


class LRTensorBoard(tf.keras.callbacks.Callback):
    # add other arguments to __init__ if you need
    def on_train_batch_end(self, batch, logs=None):
        step =self.model.optimizer.iterations.read_value()
        logs = logs or {}
        lr_schedule = getattr(self.model.optimizer, "lr", None)
        if isinstance(lr_schedule, tf.keras.optimizers.schedules.LearningRateSchedule):
            logs["lr"] = lr_schedule(self.model.optimizer.step)
        elif lr_schedule is not None:
            logs["lr"] = lr_schedule
        self.model.add_metric()
        super(LRTensorBoard,self).on_train_batch_end(batch, logs)

class LearningRate_Warmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate schedule.
        only linear warm_up
    """

    def __init__(self, end_lr, init_lr=0,  warmup_steps=4000, gradient_tower=1):
        """Initialize configuration of the learning rate schedule.
    Args:
      initial_learning_rate: A float, the initial learning rate.
      hidden_size: An integer, the model dimension in the hidden layers.
      warmup_steps: An integer, the number of steps required for linear warmup.


    """
        super(LearningRate_Warmup, self).__init__()
        self.init_lr = init_lr
        self.end_lr= end_lr
        self.warmup_steps = warmup_steps
        self.warmup_steps_tensor = tf.cast(warmup_steps, tf.float32)
        self.gradient_tower = gradient_tower

    def __call__(self, global_step):
        """Calculate learning rate with linear warmup and rsqrt decay.
    Args:
      global_step: An integer, the current global step used for learning rate
        calculation.
    Returns:
      A float, the learning rate needs to be used for current global step.
    """
        with tf.name_scope("learning_rate_schedule"):
            global_step = tf.cast(global_step, tf.float32) / self.gradient_tower
            learning_warm = self.end_lr - self.init_lr
            # Apply linear warmup
            lr= self.init_lr + tf.minimum(1.0, global_step / self.warmup_steps_tensor)*(learning_warm)
            return lr

    def get_config(self):
        """Get the configuration of the learning rate schedule."""
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "hidden_size": self.hidden_size,
            "warmup_steps": self.warmup_steps,
            "gradient_tower": self.gradient_tower,
        }

class LearningRate_WarmupRsqrtDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate schedule.
        linear warm_up + rsqrt decay
        a.k.a Transformer default
    """

    def __init__(self, initial_learning_rate, hidden_size, warmup_steps, gradient_tower=1):
        """Initialize configuration of the learning rate schedule.
    Args:
      initial_learning_rate: A float, the initial learning rate.
      hidden_size: An integer, the model dimension in the hidden layers.
      warmup_steps: An integer, the number of steps required for linear warmup.


    """
        super(LearningRate_WarmupRsqrtDecay, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.hidden_size = tf.cast(hidden_size, tf.float32)
        self.warmup_steps = warmup_steps
        self.warmup_steps_tensor = tf.cast(warmup_steps, tf.float32)
        self.gradient_tower = gradient_tower

    def __call__(self, global_step):
        """Calculate learning rate with linear warmup and rsqrt decay.
    Args:
      global_step: An integer, the current global step used for learning rate
        calculation.
    Returns:
      A float, the learning rate needs to be used for current global step.
    """
        with tf.name_scope("learning_rate_schedule"):
            global_step = tf.cast(global_step, tf.float32) / self.gradient_tower
            learning_rate = self.initial_learning_rate
            learning_rate *= self.hidden_size ** -0.5
            # Apply linear warmup
            learning_rate *= tf.minimum(1.0, global_step / self.warmup_steps_tensor)
            # Apply rsqrt decay
            learning_rate *= tf.math.rsqrt(tf.maximum(global_step, self.warmup_steps_tensor))
            return learning_rate

    def get_config(self):
        """Get the configuration of the learning rate schedule."""
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "hidden_size": self.hidden_size,
            "warmup_steps": self.warmup_steps,
            "gradient_tower": self.gradient_tower,
        }


class LearningRate_WarmupLinearDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate schedule.
        linear warm_up + linear decay
        a.k.a BERT
    """

    def __init__(self, initial_learning_rate=1e-7, end_learning_rate=1e-4, warmup_steps=10000, gradient_tower=1,exp_factor=0.5):
        """Initialize configuration of the learning rate schedule.
    Args:
      initial_learning_rate: A float, the initial learning rate.
      hidden_size: An integer, the model dimension in the hidden layers.
      warmup_steps: An integer, the number of steps required for linear warmup.
    """
        super(LearningRate_WarmupLinearDecay, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.end_learning_rate = end_learning_rate
        self.gradient_tower = gradient_tower
        # self.learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(
        #     initial_learning_rate=self.initial_learning_rate,
        #     decay_steps=self.warmup_steps,
        #     end_learning_rate=self.end_learning_rate,
        #     power=1.0,
        #     cycle=False,
        #     name=None,
        # )
        self.warmup_updates = warmup_steps
        self.warmup_init_lr = initial_learning_rate
        self.lr_step = (self.end_learning_rate - self.initial_learning_rate) / self.warmup_steps
        self.exp_factor = exp_factor
        self.decay_factor = self.end_learning_rate * self.warmup_steps ** self.exp_factor
    def __call__(self, global_step):
        """Calculate learning rate with linear warmup and rsqrt decay.
    Args:
      global_step: An integer, the current global step used for learning rate
        calculation.
    Returns:
      A float, the learning rate needs to be used for current global step.
    """
        with tf.name_scope("learning_rate_schedule"):
            # global_steps_int = tf.cast(global_step / self.gradient_tower, tf.int32)
            # warmup_steps_int = tf.constant(self.warmup_steps, dtype=tf.int32)

            # global_steps_float = tf.cast(global_steps_int, tf.float32)
            # warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

            # warmup_percent_done = global_steps_float / warmup_steps_float
            # warmup_learning_rate = self.initial_learning_rate * warmup_percent_done

            # is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
            # learning_rate = (1.0 - is_warmup) * self.learning_rate(global_steps_int) + is_warmup * warmup_learning_rate
            lr = tf.where(tf.less_equal(global_step,  self.warmup_steps),self.initial_learning_rate + global_step * self.lr_step,self.decay_factor * (global_step ** -self.exp_factor))
            # if global_step < self.warmup_steps:
            #     return self.initial_learning_rate + global_step * self.lr_step
            # else:
            #     return self.decay_factor * (global_step ** -self.exp_factor)
            return lr

    def get_config(self):
        """Get the configuration of the learning rate schedule."""
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "hidden_size": self.hidden_size,
            "warmup_steps": self.warmup_steps,
        }
