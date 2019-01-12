
import tensorflow as tf
import numpy as np
import math
from model.utils.qanet import qanet_layers

initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                             mode='FAN_AVG',
                                                             uniform=True,
                                                             dtype=tf.float32)



def hard_attention_mask(presence_vec, threshold):
    init_condition = tf.zeros_like(presence_vec)
    threshold_vec = tf.ones_like(presence_vec) * threshold
    condition = tf.less_equal(presence_vec, threshold_vec)

    default_values = tf.ones_like(presence_vec)
    mask = tf.where(condition, init_condition, default_values)
    mask = tf.cast(mask, tf.float32)
    return mask

def _position_encoding(position_size, dim, 
                    min_timescale=1.0,
                    max_timescale=1.0e4):
    position = tf.to_float(tf.range(position_size))
    num_timescales = dim // 2
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) /
        (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(
        tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) \
        * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    signal = tf.pad(signal, [[0, 0], [0, tf.mod(dim, 2)]])
    signal = tf.reshape(signal, [1, position_size, dim])

    return signal

def _add_gradient_noise(t, stddev=1e-3, name=None):
    """Adds gradient noise as described in http://arxiv.org/abs/1511.06807
    The input Tensor `t` should be a gradient.
    The output will be `t` + gaussian noise.
    0.001 was said to be a good fixed value for memory networks."""
    with tf.variable_scope('gradient_noise'):
        gn = tf.random_normal(tf.shape(t), stddev=stddev)
        return tf.add(t, gn)

def highway_layer(in_val, scope=None):
    output_size = in_val.get_shape()[-1]
    with tf.variable_scope(scope or "highway_layer"):
        highway_w = tf.get_variable("highway_w", [output_size, output_size], dtype=tf.float32)
        highway_b = tf.get_variable("highway_b", [output_size], dtype=tf.float32)
        full_w = tf.get_variable("full_w", [output_size, output_size], dtype=tf.float32)
        full_b = tf.get_variable("full_b", [output_size], dtype=tf.float32)
        trans = tf.nn.tanh(tf.nn.xw_plus_b(in_val, full_w, full_b))
        gate = tf.nn.sigmoid(tf.nn.xw_plus_b(in_val, highway_w, highway_b))
        outputs = trans*gate + in_val* (1.0- gate)
    return outputs

def multi_highway_layer(in_val, num_layers, scope=None):
    scope_name = 'highway_layer'
    if scope is not None: scope_name = scope
    for i in range(num_layers):
        cur_scope_name = scope_name + "-{}".format(i)
        in_val = highway_layer(in_val, scope=cur_scope_name)
    return in_val

def attention_bias_ignore_padding(memory_padding):
    """Create an bias tensor to be added to attention logits.

    Args:
        memory_padding: a float `Tensor` with shape [batch, memory_length].

    Returns:
        a `Tensor` with shape [batch, 1, 1, memory_length].
        each dim corresponding to batch_size, num_heads, queries_len,
        memory_length
    """
    ret = memory_padding * -1e18
    return tf.expand_dims(tf.expand_dims(ret, axis=1), axis=1)

def _split_heads(x, num_heads):
    """Split channels (dimension 2) into multiple heads,
        becomes dimension 1).
    Must ensure `x.shape[-1]` can be deviced by num_heads
    """
    depth = x.get_shape()[-1]
    print(x.get_shape(), "===splitheads===")
    splitted_x = tf.reshape(x, [tf.shape(x)[0], tf.shape(x)[1], \
        num_heads, depth // num_heads])
    return tf.transpose(splitted_x, [0, 2, 1, 3])

def _combine_heads(x):
    """
    Args:
        x: A Tensor of shape `[batch, num_heads, seq_len, dim]`

    Returns:
        A Tensor of shape `[batch, seq_len, num_heads * dim]`
    """
    t = tf.transpose(x, [0, 2, 1, 3]) #[batch, seq_len, num_heads, dim]
    num_heads, dim = t.get_shape()[-2:]
    return tf.reshape(t, [tf.shape(t)[0], tf.shape(t)[1], num_heads*dim])

def multihead_attention_texar(queries, 
                memory=None, 
                memory_attention_bias=None,
                num_heads=8, 
                num_units=None, 
                dropout_rate=0.0, 
                scope="multihead_attention"):
    if num_units is None:
        num_units = queries.get_shape()[-1]
    if num_units % num_heads != 0:
            raise ValueError("Value depth (%d) must be divisible by the"
                             "number of attention heads (%d)." % (\
                            num_units, num_heads))
    if memory is None:
        Q = tf.layers.dense(queries, num_units, use_bias=False, name='q')
        K = tf.layers.dense(queries, num_units, use_bias=False, name='k')
        V = tf.layers.dense(queries, num_units, use_bias=False, name='v')
    else:
        Q = tf.layers.dense(queries, num_units, use_bias=False, name='q')
        K = tf.layers.dense(memory, num_units, use_bias=False, name='k')
        V = tf.layers.dense(memory, num_units, use_bias=False, name='v')

    Q_ = _split_heads(Q, num_heads)
    K_ = _split_heads(K, num_heads)
    V_ = _split_heads(V, num_heads)

    key_depth_per_head = num_units // num_heads
    Q_ *= tf.pow(tf.cast(key_depth_per_head, tf.float32), -0.5)

    logits = tf.matmul(Q_, K_, transpose_b=True)
    if memory_attention_bias is not None:
        logits += memory_attention_bias
    weights = tf.nn.softmax(logits, name="attention_weights")
    weights = tf.nn.dropout(weights, 1 - dropout_rate)
    outputs = tf.matmul(weights, V_)

    outputs = _combine_heads(outputs)
    outputs = tf.layers.dense(outputs, num_units,\
            use_bias=False, name='output_transform')
        #(batch_size, length_query, attention_depth)
    return outputs

def last_relevant_output(output, sequence_length):
    """
    Given the outputs of a LSTM, get the last relevant output that
    is not padding. We assume that the last 2 dimensions of the input
    represent (sequence_length, hidden_size).

    Parameters
    ----------
    output: Tensor
        A tensor, generally the output of a tensorflow RNN.
        The tensor index sequence_lengths+1 is selected for each
        instance in the output.

    sequence_length: Tensor
        A tensor of dimension (batch_size, ) indicating the length
        of the sequences before padding was applied.

    Returns
    -------
    last_relevant_output: Tensor
        The last relevant output (last element of the sequence), as retrieved
        by the output Tensor and indicated by the sequence_length Tensor.
    """
    with tf.name_scope("last_relevant_output"):
        batch_size = tf.shape(output)[0]
        max_length = tf.shape(output)[-2]
        out_size = int(output.get_shape()[-1])
        index = tf.range(0, batch_size) * max_length + (sequence_length - 1)
        flat = tf.reshape(output, [-1, out_size])
        relevant = tf.gather(flat, index)
        return relevant

def hard_attention(context, context_mask,
                    num_heads, dropout_rate):

    print(context.get_shape(), "==hard attention shape==")

    preserve_vec = context*context
    preserve_vec = tf.sqrt(tf.reduce_sum(preserve_vec, axis=-1))
    
    preserve_vec = qanet_layers.mask_logits(preserve_vec, context_mask)
    preserve_norm = tf.nn.softmax(preserve_vec)

    print(preserve_norm.get_shape(), "==preserve_norm shape==")
    threshold = 1 / tf.reduce_sum(
                        tf.cast(context_mask, tf.float32), 
                        axis=-1,
                        keep_dims=True)

    preserve_mask = hard_attention_mask(preserve_norm, threshold)

    print(preserve_mask.get_shape(), "==preserve_mask shape==")

    context *= tf.expand_dims(preserve_mask, -1)
    ignore_padding = (1 - preserve_mask)
    ignore_padding = attention_bias_ignore_padding(ignore_padding)
    encoder_self_attention_bias = ignore_padding

    output = multihead_attention_texar(context, 
                    memory=None, 
                    memory_attention_bias=encoder_self_attention_bias,
                    num_heads=num_heads, 
                    num_units=None, 
                    dropout_rate=dropout_rate, 
                    scope="multihead_attention")
    output = tf.reduce_sum(output, axis=1)
    return output

def alignment_hard_attention(query,
                    context, context_mask,
                    num_heads, dropout_rate):

    preserve_vec = tf.einsum("ac,abc->ab", query, context) # batch x context_len
    preserve_norm = tf.nn.softmax(qanet_layers.mask_logits(preserve_vec, mask = context_mask))

    # batch x 1
    threshold = 1 / tf.reduce_sum(
                            tf.cast(context_mask, tf.float32), 
                            axis=-1,
                            keep_dims=True)

    preserve_mask = hard_attention_mask(preserve_norm, threshold)
    context *= tf.expand_dims(preserve_mask, -1)

    ignore_padding = (1 - preserve_mask)
    ignore_padding = attention_bias_ignore_padding(ignore_padding)
    encoder_self_attention_bias = ignore_padding

    output = multihead_attention_texar(context, 
                    memory=None, 
                    memory_attention_bias=encoder_self_attention_bias,
                    num_heads=num_heads, 
                    num_units=None, 
                    dropout_rate=dropout_rate, 
                    scope="multihead_attention")
    output = tf.reduce_sum(output, axis=1)

    return output



    