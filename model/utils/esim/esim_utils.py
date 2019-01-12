import tensorflow as tf
import numpy as np
from model.utils.qanet import qanet_layers
from model.utils.man import man_utils
import math

initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                             mode='FAN_AVG',
                                                             uniform=True,
                                                             dtype=tf.float32)

def query_context_attention(query, context, max_query_len, max_context_len, 
                query_mask, context_mask, dropout_ratio,
                scope, reuse=None):
    with tf.variable_scope(scope+"_Context_to_Query_Attention_Layer", reuse=reuse):
        context_ = tf.transpose(context, [0,2,1])
        hiddem_dim = query.get_shape()[-1]

        attn_W = tf.get_variable("AttnW", dtype=tf.float32,
                                    shape=[hiddem_dim, hiddem_dim],
                                    initializer=initializer)

        weighted_query = tf.tensordot(query, attn_W, axes=[[2], [0]])

        S = tf.matmul(weighted_query, context_)  # batch x q_len x c_len

        mask_q = tf.expand_dims(query_mask, 1)
        mask_c = tf.expand_dims(context_mask, 1)

        S_ = tf.nn.softmax(qanet_layers.mask_logits(S, mask = mask_c))
        c2q = tf.matmul(S_, context)

        S_T = tf.nn.softmax(qanet_layers.mask_logits(tf.transpose(S, [0,2,1]), mask = mask_q))
        q2c = tf.matmul(S_T, query)

        query_attention_outputs = tf.concat([query, c2q, query-c2q, query*c2q], axis=-1)
        query_attention_outputs *= tf.expand_dims(tf.cast(query_mask, tf.float32), -1)

        context_attention_outputs = tf.concat([context, q2c, context-q2c, context*q2c], axis=-1)
        context_attention_outputs *= tf.expand_dims(tf.cast(context_mask, tf.float32), -1)

        return query_attention_outputs, context_attention_outputs

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


