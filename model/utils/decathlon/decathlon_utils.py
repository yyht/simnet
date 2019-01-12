import tensorflow as tf
import numpy as np
from model.utils.qanet import qanet_layers
from model.utils.man import man_utils
import math

initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                             mode='FAN_AVG',
                                                             uniform=True,
                                                             dtype=tf.float32)

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

def multihead_attention(queries, 
                        keys, 
                        num_units=None, 
                        num_heads=8, 
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="multihead_attention", 
                        reuse=None):
    '''Applies multihead attention.
    
    Args:
        queries: A 3d tensor with shape of [N, T_q, C_q].
        keys: A 3d tensor with shape of [N, T_k, C_k].
        num_units: A scalar. Attention size.
        dropout_rate: A floating point number.
        is_training: Boolean. Controller of mechanism for dropout.
        causality: Boolean. If true, units that reference the future are masked. 
        num_heads: An int. Number of heads.
        scope: Optional scope for `variable_scope`.
        reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.
            
    Returns
        A 3d tensor with shape of (N, T_q, C)   
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape()[-1]
        
        # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu) # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
        
        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, C/h) 
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # (h*N, T_q, T_k)
        
        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
        
        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1))) # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1]) # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]) # (h*N, T_q, T_k)
        
        paddings = tf.ones_like(outputs)*(-2**32+1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k)

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :]) # (T_q, T_k)
            tril = tf.contrib.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense() # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1]) # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks)*(-2**32+1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs) # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs) # (h*N, T_q, T_k)
         
        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1))) # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1]) # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]) # (h*N, T_q, T_k)
        outputs *= query_masks # broadcasting. (N, T_q, C)
            
        # Dropouts
        outputs = tf.nn.dropout(outputs, 1 - dropout_rate)
                     
        # Weighted sum
        outputs = tf.matmul(outputs, V_) # ( h*N, T_q, C/h)
        
        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # (N, T_q, C)
                    
        # Residual connection
        # outputs += queries
                    
        # Normalize
        # outputs = tf.contrib.layers.layer_norm(outputs)

        # outputs = normalize(outputs) # (N, T_q, C)
        # outputs = tf.cast(outputs, dtype=tf.float32)

    return outputs

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


def interaction_attention(query, context, 
                        query_mask, context_mask, 
                        attn_lst, dropout_ratio, scope):
    q2c, c2q = [], [] # q2c=context, c2a=query
    for attn in attn_lst:
        if attn == "concat":
            [c2q_concat, 
            q2c_concat] = man_utils.concat_attention(query, context,
                                        query_mask, context_mask, dropout_rate,
                                        scope, reuse=reuse)
            q2c.append(q2c_concat)
            c2q.append(c2q_concat)
        elif attn == "bilinear"

            [c2q_bilinear, 
            q2c_bilinear] = man_utils.bilinear_attention(query, context,
                                        query_mask, context_mask, dropout_rate,
                                        scope, reuse=reuse)
            q2c.append(q2c_bilinear)
            c2q.append(c2q_bilinear)
        elif attn == "dot":
            [c2q_dot, 
            q2c_dot] = man_utils.dot_attention(query, context,
                                        query_mask, context_mask, dropout_rate,
                                        scope, reuse=reuse)
            q2c.append(q2c_dot)
            c2q.append(c2q_dot)
        elif attn == "minus":
            [c2q_minus, 
            q2c_minus] = man_utils.minus_attention(query, context,
                                        query_mask, context_mask, dropout_rate,
                                        scope, reuse=reuse)

            q2c.append(q2c_minus)
            c2q.append(c2q_minus)

    return c2q, q2c

def attention_fusion(context, context_mask, 
                    scope, reuse):

    with tf.variable_scope(scope, reuse=reuse):
        # query size  batch x 4 x len x dim

        hidden_dim = context.get_shape()[-1]
        repres = context
        repres_mask = context_mask

        W = tf.get_variable(scope+"_W", 
                            dtype=tf.float32,
                            shape=[hidden_dim, hidden_dim],
                            initializer=initializer)
        B = tf.get_variable(scope+"_B", 
                            dtype=tf.float32,
                            shape=[hidden_dim],
                            initializer=initializer)
        V = tf.get_variable(scope+"_V", 
                            dtype=tf.float32,
                            shape=[hidden_dim],
                            initializer=initializer)

        # batch x 4 x len x dim
        project = tf.einsum("abcd,de->adce", repres, W) + B
        project = tf.nn.tanh(project)

        S = tf.einsum("abcd,d->abc", project, V) # batch x 4 x len

        S_ = tf.nn.softmax(S, axis=1) # batch x 4 x len
        S_ = tf.expand_dims(S_, axis=-1) # batch x 4 x len x 1

        repres = tf.reduce_sum(S_ * project, axis=1) # batch x len x dim
        return repres





