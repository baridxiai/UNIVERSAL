# -*- coding: utf-8 -*-
# code warrior: Barid
import tensorflow as tf
from tensorflow.python.training import training_ops
from tensorflow.python.ops import math_ops

# import runai.ga.keras
from runai.ga.keras import hooks
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


class MultistepAdamOptimizer(tf.compat.v1.train.AdamOptimizer):
    """Adam with SGD updates every n steps with accumulated gradients."""

    def __init__(
        self,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        use_locking=False,
        name="Adam",
        warmmup_steps=4000,
        d_model=512,
        n=2,
    ):
        super(MultistepAdamOptimizer, self).__init__(
            learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon, use_locking=use_locking, name=name
        )
        self._n = n  # Call Adam optimizer every n batches with accumulated grads
        self._n_t = None  # n as tensor
        self.lr = tf.cast(learning_rate, tf.float32)
        self.constant_d_model_tensor = tf.cast(d_model, tf.float32)
        self.constant_minus05_tensor = tf.cast(-0.5, tf.float32)
        self.constant_minus15_tensor = tf.cast(-1.5, tf.float32)
        self.constant_warmup_tensor = tf.cast(warmmup_steps, tf.float32)
        self.beta1 = beta1
        self.beta2 = beta2

    def _create_slots(self, var_list):
        """Create slot variables for Adam with accumulated gradients."""
        # Create the beta1 and beta2 accumulators, and lambda_update on the same device as the first
        # variable. Sort the var_list to make sure this device is consistent across
        # workers (these need to go on the same PS, otherwise some updates are
        # silently ignored).
        first_var = min(var_list, key=lambda x: x.name)
        self._create_non_slot_variable(initial_value=self.beta1, name="beta1_power", colocate_with=first_var)
        self._create_non_slot_variable(initial_value=self.beta2, name="beta2_power", colocate_with=first_var)
        self._create_non_slot_variable(initial_value=0.00001, name="lambda_update", colocate_with=first_var)
        self._create_non_slot_variable(initial_value=0 if self._n == 1 else 1, name="iter", colocate_with=first_var)
        # Create slots for the first and second moments, as well as grad_acc.
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)
            self._zeros_slot(v, "grad_acc", self._name)

    def _get_iter_variable(self):
        graph = None if tf.executing_eagerly() else tf.compat.v1.get_default_graph()
        return self._get_non_slot_variable("iter", graph=graph)

    def _prepare(self):
        super(MultistepAdamOptimizer, self)._prepare()
        self._n_t = tf.convert_to_tensor(self._n, name="n")

    def _apply_cond(self, apply_fn, grad, var, *args, **kwargs):
        """Apply conditionally if counter is zero."""
        grad_acc = self.get_slot(var, "grad_acc")

        def apply_adam(grad_acc, apply_fn, grad, var, *args, **kwargs):
            total_grad = (grad_acc + grad) / tf.cast(self._n_t, grad.dtype)
            adam_op = apply_fn(total_grad, var, *args, **kwargs)
            with tf.control_dependencies([adam_op]):
                grad_acc_to_zero_op = grad_acc.assign(tf.zeros_like(grad_acc), use_locking=self._use_locking)
            return tf.group(adam_op, grad_acc_to_zero_op)

        def accumulate_gradient(grad_acc, grad):
            assign_op = tf.assign_add(grad_acc, grad, use_locking=self._use_locking)
            return tf.group(assign_op)  # Strip return value

        return tf.cond(
            tf.equal(self._get_iter_variable(), 0),
            lambda: apply_adam(grad_acc, apply_fn, grad, var, *args, **kwargs),
            lambda: accumulate_gradient(grad_acc, grad),
        )

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        return self._apply_cond(
            super(MultistepAdamOptimizer, self)._apply_sparse_shared, grad, var, indices, scatter_add
        )

    def _apply_sparse(self, grad, var):
        # TODO(fstahlberg): Implement a sparse version
        tf.logging.warning("MultistepAdamOptimizer does not support sparse updates")
        dense_grad = tf.convert_to_tensor(grad)
        return self._apply_cond(self._apply_dense, dense_grad, var)

    def _resource_apply_sparse_duplicate_indices(self, grad, var, indices):
        tf.logging.warning("MultistepAdamOptimizer does not support sparse updates")
        # Note that conversion to a dense Tensor handles duplicate `indices`
        # correctly (summing them). A real sparse implementation will probably want
        # to override _resource_apply_sparse instead so it gets them de-duplicated
        # automatically.
        dense_grad = tf.convert_to_tensor(tf.IndexedSlices(values=grad, indices=indices, dense_shape=tf.shape(var)))
        return self._apply_cond(self._resource_apply_dense, dense_grad, var)

    def _apply_dense(self, grad, var):
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        beta1_power, beta2_power, lambda_update = self._get_beta_lamda_accumulators()
        return training_ops.apply_adam(
            var,
            m,
            v,
            math_ops.cast(beta1_power, var.dtype.base_dtype),
            math_ops.cast(beta2_power, var.dtype.base_dtype),
            math_ops.cast(lambda_update, var.dtype.base_dtype),
            math_ops.cast(self._beta1_t, var.dtype.base_dtype),
            math_ops.cast(self._beta2_t, var.dtype.base_dtype),
            math_ops.cast(self._epsilon_t, var.dtype.base_dtype),
            grad,
            use_locking=self._use_locking,
        ).op

    def _resource_apply_dense(self, grad, var):
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        beta1_power, beta2_power, lambda_update = self._get_beta_lamda_accumulators()
        return training_ops.resource_apply_adam(
            var.handle,
            m.handle,
            v.handle,
            tf.cast(beta1_power, grad.dtype.base_dtype),
            tf.cast(beta2_power, grad.dtype.base_dtype),
            tf.cast(lambda_update, grad.dtype.base_dtype),
            tf.cast(self._beta1_t, grad.dtype.base_dtype),
            tf.cast(self._beta2_t, grad.dtype.base_dtype),
            tf.cast(self._epsilon_t, grad.dtype.base_dtype),
            grad,
            use_locking=self._use_locking,
        )

    def _get_beta_lamda_accumulators(self):
        with tf.init_scope():
            graph = None if tf.executing_eagerly() else tf.compat.v1.get_default_graph()
            return (
                self._get_non_slot_variable("beta1_power", graph=graph),
                self._get_non_slot_variable("beta2_power", graph=graph),
                self._get_non_slot_variable("lambda_update", graph=graph),
            )

    def _finish(self, update_ops, name_scope):
        # Trasnformer: constant_d_model_tensor**(-0.5)*min(global_step**(-0.5), global_step*(num_warmup_steps**(-1.5)))
        """Updates beta_power variables every n batches and incrs counter."""

        with tf.init_scope():
            iter_ = self._get_iter_variable()
        constant_iter_plus = tf.cast(iter_ + 1, tf.float32)  # constant_iter_plus = iter_+ plus 1
        # print("iter", iter)
        beta1_power, beta2_power, lambda_update = self._get_beta_lamda_accumulators()
        with tf.control_dependencies(update_ops):
            with tf.compat.v1.colocate_with(iter_):

                def update_beta_op():
                    update_beta1 = beta1_power.assign(beta1_power * self._beta1_t, use_locking=self._use_locking)
                    update_beta2 = beta2_power.assign(beta2_power * self._beta2_t, use_locking=self._use_locking)
                    lr = self.lr * tf.math.pow(self.constant_d_model_tensor, self.constant_minus05_tensor)
                    lr *= tf.minimum(1.0, constant_iter_plus / self.constant_warmup_tensor)
                    lr *= tf.math.rsqrt(tf.maximum(constant_iter_plus, self.constant_warmup_tensor))
                    update_lambda_update = lambda_update.assign(lr, use_locking=self._use_locking,)
                    return tf.group(update_beta1, update_beta2, update_lambda_update)

                maybe_update_beta = tf.cond(tf.equal(tf.math.mod(iter_ + 1, self._n_t), 0), update_beta_op, tf.no_op)
                with tf.control_dependencies([maybe_update_beta]):
                    update_iter = iter_.assign(iter_ + 1, use_locking=self._use_locking)
        return tf.group(*update_ops + [update_iter, maybe_update_beta], name=name_scope)


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
        super(AdamWeightDecay, self)._prepare_local(var_device, var_dtype, apply_state)
        apply_state["weight_decay_rate"] = tf.constant(self.weight_decay_rate, name="adam_weight_decay_rate")

    def _decay_weights_op(self, var, learning_rate, apply_state):
        do_decay = self._do_use_weight_decay(var.name)
        if do_decay:
            return var.assign_sub(
                learning_rate * var * apply_state["weight_decay_rate"], use_locking=self._use_locking,
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

        apply_state = apply_state or {}
        coefficients = apply_state.get((var_device, var_dtype))
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
