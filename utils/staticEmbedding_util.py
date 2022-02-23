# -*- coding: utf-8 -*-
# code warrior: Barid
import math
import tensorflow as tf
from UNIVERSAL.utils import padding_util


def get_position_encoding(length, hidden_size, min_timescale=1.0, max_timescale=1.0e4):
    """Return positional encoding.
  Calculates the position encoding as a mix of sine and cosine functions with
  geometrically increasing wavelengths.
  Defined and formulized in Attention is All You Need, section 3.5.
  Args:
    length: Sequence length.
    hidden_size: Size of the
    min_timescale: Minimum scale that will be applied at each position
    max_timescale: Maximum scale that will be applied at each position
  Returns:
    Tensor with shape [length, hidden_size]
  """
    position = tf.cast(tf.range(length), tf.float32)
    num_timescales = hidden_size // 2
    log_timescale_increment = math.log(float(max_timescale) / float(min_timescale)) / (
        tf.cast(num_timescales, tf.float32) - 1
    )
    inv_timescales = min_timescale * tf.exp(tf.cast(tf.range(num_timescales), tf.float32) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    return signal


def get_masked_position_encoding(x, hidden_size, min_timescale=1.0, max_timescale=1.0e4):
    """Return positional encoding.
  Calculates the position encoding as a mix of sine and cosine functions with
  geometrically increasing wavelengths.
  Defined and formulized in Attention is All You Need, section 3.5.
  Args:
    length: Sequence length.
    hidden_size: Size of the
    min_timescale: Minimum scale that will be applied at each position
    max_timescale: Maximum scale that will be applied at each position
  Returns:
    Tensor with shape [length, hidden_size]
  """
    length = tf.shape(x)[1]
    batch_size = tf.shape(x)[2]
    signal = get_position_encoding(length, hidden_size, min_timescale, max_timescale)
    signal = tf.broadcast_to(signal, shape=[batch_size] + signal.get_shape().as_list())
    mask = tf.expand_dims(1 - padding_util.get_padding(x), -1)
    signal = signal * mask

    # if mask is not None:

    return signal


def get_non_value_position(x, value=0):
    bool_mask = tf.cast(tf.not_equal(x, value), tf.int32)
    length = tf.shape(x)[1]
    batch_size = tf.shape(x)[0]
    position = tf.range(length)
    position = tf.broadcast_to(position, shape=[batch_size] + position.get_shape())
    return tf.boolean_mask(position, bool_mask)


def get_value_position(x, value=0):
    bool_mask = tf.cast(tf.equal(x, value), tf.int32)
    test = tf.reduce_sum(bool_mask, -1)
    max_value = tf.reduce_max(test)
    test = max_value - test
    bool_mask_1 = tf.concat((bool_mask[:, :-2], tf.expand_dims(test, -1)), -1)
    bool_mask_1 = tf.concat((bool_mask_1, tf.expand_dims(test, -1)), -1)
    bool_mask = bool_mask_1 + bool_mask
    bool_mask = tf.cast(tf.greater(bool_mask, 0), tf.int32)
    length = tf.shape(x)[1]
    batch_size = tf.shape(x)[0]
    position = tf.range(length)
    position = tf.broadcast_to(position, shape=[batch_size, length])
    return tf.reshape(tf.boolean_mask(position, bool_mask), shape=[batch_size, -1])


def get_timing_signal_1d(length, channels, min_timescale=1.0, max_timescale=1.0e4, start_index=0):
    """Gets a bunch of sinusoids of different frequencies.
    Each channel of the input Tensor is incremented by a sinusoid of a different
    frequency and phase.
    This allows attention to learn to use absolute and relative positions.
    Timing signals should be added to some precursors of both the query and the
    memory inputs to attention.
    The use of relative position is possible because sin(x+y) and cos(x+y) can be
    expressed in terms of y, sin(x) and cos(x).
    In particular, we use a geometric sequence of timescales starting with
    min_timescale and ending with max_timescale.  The number of different
    timescales is equal to channels / 2. For each timescale, we
    generate the two sinusoidal signals sin(timestep/timescale) and
    cos(timestep/timescale).  All of these sinusoids are concatenated in
    the channels dimension.
    Args:
      length: scalar, length of timing signal sequence.
      channels: scalar, size of timing embeddings to create. The number of
          different timescales is equal to channels / 2.
      min_timescale: a float
      max_timescale: a float
      start_index: index of first position
    Returns:
      a Tensor of timing signals [1, length, channels]
    """
    position = tf.cast(tf.range(length) + start_index, tf.float32)
    num_timescales = channels // 2
    log_timescale_increment = math.log(float(max_timescale) / float(min_timescale)) / tf.maximum(
        tf.cast(num_timescales, tf.float32) - 1, 1
    )
    inv_timescales = min_timescale * tf.exp(tf.cast(tf.range(num_timescales), tf.float32) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    # Please note that this slightly differs from the published paper.
    # See a discussion here: https://github.com/tensorflow/tensor2tensor/pull/177
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    signal = tf.pad(signal, [[0, 0], [0, channels % 2]])
    signal = tf.reshape(signal, [1, length, channels])
    return signal


def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4, start_index=0):
    """Adds a bunch of sinusoids of different frequencies to a Tensor.
    Each channel of the input Tensor is incremented by a sinusoid of a different
    frequency and phase.
    This allows attention to learn to use absolute and relative positions.
    Timing signals should be added to some precursors of both the query and the
    memory inputs to attention.
    The use of relative position is possible because sin(x+y) and cos(x+y) can be
    expressed in terms of y, sin(x) and cos(x).
    In particular, we use a geometric sequence of timescales starting with
    min_timescale and ending with max_timescale.  The number of different
    timescales is equal to channels / 2. For each timescale, we
    generate the two sinusoidal signals sin(timestep/timescale) and
    cos(timestep/timescale).  All of these sinusoids are concatenated in
    the channels dimension.
    Args:
      x: a Tensor with shape [batch, length, channels]
      min_timescale: a float
      max_timescale: a float
      start_index: index of first position
    Returns:
      a Tensor the same shape as x.
    """
    length = tf.shape(x)[1]
    channels = tf.shape(x)[2]
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale, start_index)
    return x + tf.cast(signal, x.dtype)


def add_position_timing_signal(x, shift, position=None, length=None):
    """Add n-dimensional embedding as the position (horizontal) timing signal.
    Args:
    x: a tensor with shape [batch, length, depth]
    step: step
    hparams: model hyper parameters
    Returns:
    a Tensor with the same shape as x.
    """
    if position is not None:
        assert length is not None, "Requiring manual lenght setting for position mode."
    if length is None:
        length = tf.shape(x)[1]
    channels = tf.shape(x)[2]
    index = tf.cast(shift, dtype=tf.int32)
    signal = get_timing_signal_1d(length, channels, start_index=index)
    if position is not None:
        # if position > 1e8:
        #     x_with_timing = x
        # else:
        # x_with_timing = x + position
        if position == -1:
            x_with_timing = x
        else:
            x_with_timing = x + tf.expand_dims(tf.cast(signal, x.dtype)[:, position, :], 1)
    else:
        x_with_timing = x + tf.cast(signal, x.dtype)

    return x_with_timing


def add_span_position_timing_signal(x, step, num_steps, span=[[0, 1]]):
    """Add n-dimensional embedding as the position (horizontal) timing signal.
    Args:
    x: a tensor with shape [batch, length, depth]
    step: step
    hparams: model hyper parameters
    Returns:
    a Tensor with the same shape as x.
    """

    length = tf.shape(x)[1]
    channels = tf.shape(x)[2]
    index = tf.cast(length * step / num_steps, dtype=tf.int32)
    signal = tf.squeeze(get_timing_signal_1d(length, channels, start_index=index), axis=0)
    x_with_timing = tf.gather(signal, span)

    return x_with_timing


def get_step_timing_signal_sinusoid_1d(channels, step, num_steps):
    """Add sinusoids of different frequencies as layer (vertical) timing signal.
    Args:
    channels: dimension of the timing signal
    layer: layer num
    num_layers: total number of layers
    Returns:
    a Tensor of timing signals [1, 1, channels].
    """
    signal = get_timing_signal_1d(num_steps, channels)
    layer_signal = tf.expand_dims(signal[:, step, :], axis=1)

    return layer_signal


def add_step_timing_signal(x, step, num_steps=None):
    """Add n-dimensional embedding as the step (vertical) timing signal.
    Args:
    x: a tensor with shape [batch, length, depth]
    step: step
    num_steps: behaviour control
    Returns:
    a Tensor with the same shape as x.

    The default of num_steps is 50.
    If num_steps =-1, it means no need step signal, then returning x
    """
    channels = tf.shape(x)[-1]
    if num_steps is None:
        num_steps = 50
    if num_steps == -1:
        return x
    signal = get_step_timing_signal_sinusoid_1d(channels, step, num_steps)
    x_with_timing = x + tf.cast(signal, x.dtype)

    return x_with_timing


def add_timing_signals_given_positions(x, positions, min_timescale=1.0, max_timescale=1.0e4):
    """Adds sinusoids of diff frequencies to a Tensor, with timing positions given.
  Args:
    x: a Tensor with shape [batch, length, channels]
    positions: a list of positions, each of which can either be a Tensor of
      shape [batch, length] or None for a default of (0..length]
    min_timescale: a float
    max_timescale: a float
  Returns:
    a Tensor the same shape as x.
  """
    shape = tf.shape(x)
    batch = shape[0]
    length = shape[1]
    channels = shape[2]
    num_dims = len(positions)
    num_timescales = channels // (num_dims * 2)
    log_timescale_increment = math.log(float(max_timescale) / float(min_timescale)) / (tf.to_float(num_timescales) - 1)
    inv_timescales = min_timescale * tf.exp(tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    for dim, position in enumerate(positions):
        if position is None:
            # Create a [batch, length] Tensor of incrementing positions 0..length-1.
            position = tf.tile(tf.transpose(tf.expand_dims(tf.range(0, length), axis=1)), [batch, 1])
        scaled_time = tf.expand_dims(tf.to_float(position), 2) * tf.expand_dims(tf.expand_dims(inv_timescales, 0), 0)
        signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=2)
        prepad = dim * 2 * num_timescales
        postpad = channels - (dim + 1) * 2 * num_timescales
        signal = tf.pad(signal, [[0, 0], [0, 0], [prepad, postpad]])
        signal = tf.cast(signal, x.dtype)
        x += signal
    return x


def add_timing_signals_from_features(x, features, position_features, min_timescale=1.0, max_timescale=1.0e4):
    """Adds timing signals from features named in `position_features`.
  Args:
    x: a Tensor with shape [batch, length, channels]
    features: a features dictionary
    position_features: a comma-delimited string where each item is either a
      feature key or the empty string (which denotes a default position tensor
      of [0..length])
    min_timescale: a float
    max_timescale: a float
  Returns:
    a Tensor the same shape as x.
  """
    return add_timing_signals_given_positions(
        x,
        [features.get(position_feature) for position_feature in position_features.split(",")],
        min_timescale,
        max_timescale,
    )
