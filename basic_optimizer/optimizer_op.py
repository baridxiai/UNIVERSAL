# -*- coding: utf-8 -*-
# code warrior: Barid
import tensorflow as tf
from tensorflow.python.training import training_ops
from tensorflow.python.ops import math_ops

# import runai.ga.keras
import re


def configure_optimizer(optimizer, use_float16=False, loss_scale=None):
    """Configures optimizer object with performance options."""
    if use_float16:
        if loss_scale in (None, "dynamic"):
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        else:
            # loss_scale is a number. We interpret that as a fixed loss scale.
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer, dynamic=False, initial_scale=loss_scale)
    return optimizer


class AdamWeightDecay(tf.keras.optimizers.Adam):
    """Adam enables L2 weight decay and clip_by_global_norm on gradients.

  Just adding the square of the weights to the loss function is *not* the
  correct way of using L2 regularization/weight decay with Adam, since that will
  interact with the m and v parameters in strange ways.

  Instead we want ot decay the weights in a manner that doesn't interact with
  the m/v parameters. This is equivalent to adding the square of the weights to
  the loss with plain (non-momentum) SGD.
  """

    def __init__(
        self,
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        amsgrad=False,
        weight_decay_rate=0.01,
        include_in_weight_decay=None,
        exclude_from_weight_decay=None,
        # clipnorm=1.0,
        name="AdamWeightDecay",
        **kwargs
    ):
        super(AdamWeightDecay, self).__init__(learning_rate, beta_1, beta_2, epsilon, amsgrad, name, **kwargs)
        self.weight_decay_rate = weight_decay_rate
        self._include_in_weight_decay = include_in_weight_decay
        self._exclude_from_weight_decay = exclude_from_weight_decay

    # @classmethod
    # def from_config(cls, config):
    #     """Creates an optimizer from its config with WarmUp custom object."""
    #     custom_objects = {"WarmUp": warmup_lr}
    #     return super(AdamWeightDecay, cls).from_config(
    #         config, custom_objects=custom_objects
    #     )

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(tf.keras.optimizers.Adam, self)._prepare_local(var_device, var_dtype, apply_state)
        # apply_state["weight_decay_rate"] = tf.constant(self.weight_decay_rate, name="adam_weight_decay_rate")

        local_step = tf.cast(self.iterations + 1, var_dtype)
        beta_1_t = tf.identity(self._get_hyper('beta_1', var_dtype))
        beta_2_t = tf.identity(self._get_hyper('beta_2', var_dtype))
        beta_1_power = tf.pow(beta_1_t, local_step)
        beta_2_power = tf.pow(beta_2_t, local_step)
        lr = (apply_state[(var_device, var_dtype)]['lr_t'] *
            (tf.sqrt(1 - beta_2_power) / (1 - beta_1_power)))
        apply_state[(var_device, var_dtype)].update(
        dict(
            lr=lr,
            epsilon=tf.convert_to_tensor(
                self.epsilon, var_dtype),
            beta_1_t=beta_1_t,
            beta_1_power=beta_1_power,
            one_minus_beta_1_t=1 - beta_1_t,
            beta_2_t=beta_2_t,
            beta_2_power=beta_2_power,
            one_minus_beta_2_t=1 - beta_2_t,
            weight_decay_rate=tf.constant(self.weight_decay_rate, name="adam_weight_decay_rate")))
    def _decay_weights_op(self, var, learning_rate, apply_state):
        do_decay = self._do_use_weight_decay(var.name)
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                    or self._fallback_apply_state(var_device, var_dtype))
        if do_decay:
            return var.assign_sub(
                learning_rate * var * coefficients["weight_decay_rate"], use_locking=self._use_locking,
            )
        return tf.no_op()

    # def apply_gradients(self, grads_and_vars, name=None):
    #     grads, tvars = list(zip(*grads_and_vars))
    #     (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
    #     return super(AdamWeightDecay, self).apply_gradients(zip(grads, tvars))

    def _get_lr(self, var_device, var_dtype, apply_state):
        """Retrieves the learning rate with the given state."""
        if apply_state is None:
            return self._decayed_lr_t[var_dtype], {}

        coefficients = (apply_state or {}).get((var_device, var_dtype)) or self._fallback_apply_state(
            var_device, var_dtype
        )
        if coefficients is None:
            coefficients = self._fallback_apply_state(var_device, var_dtype)
            apply_state[(var_device, var_dtype)] = coefficients

        return coefficients["lr_t"], dict(apply_state=apply_state)

    def _resource_apply_dense(self, grad, var, apply_state=None):
        lr_t, kwargs = self._get_lr(var.device, var.dtype.base_dtype, apply_state)
        decay = self._decay_weights_op(var, lr_t, apply_state)
        with tf.control_dependencies([decay]):
            return super(AdamWeightDecay, self)._resource_apply_dense(grad, var, **kwargs)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        lr_t, kwargs = self._get_lr(var.device, var.dtype.base_dtype, apply_state)
        decay = self._decay_weights_op(var, lr_t, apply_state)
        with tf.control_dependencies([decay]):
            return super(AdamWeightDecay, self)._resource_apply_sparse(grad, var, indices, **kwargs)

    def get_config(self):
        config = super(AdamWeightDecay, self).get_config()
        config.update(
            {"weight_decay_rate": self.weight_decay_rate,}
        )
        return config

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if self.weight_decay_rate == 0:
            return False

        if self._include_in_weight_decay:
            for r in self._include_in_weight_decay:
                if re.search(r, param_name) is not None:
                    return True

        if self._exclude_from_weight_decay:
            for r in self._exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True


class AdamWeightDecayLinearDecay(AdamWeightDecay):
    def __init__(
        self,
        initial_learning_rate=1e-7,
        end_learning_rate=1e-4,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        num_warmup_steps=30000,
        amsgrad=False,
        weight_decay_rate=0.01,
        include_in_weight_decay=None,
        exclude_from_weight_decay=None,
        # clipnorm=1.0,
        name="AdamWeightDecayLinearDecay",
        **kwargs
    ):
        learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=num_warmup_steps,
            end_learning_rate=end_learning_rate,
            power=1.0,
            cycle=False,
            name=None,
        )
        super(AdamWeightDecay, self).__init__(learning_rate, beta_1, beta_2, epsilon, amsgrad, name, **kwargs)
        self.weight_decay_rate = weight_decay_rate
        self._include_in_weight_decay = include_in_weight_decay
        self._exclude_from_weight_decay = exclude_from_weight_decay
