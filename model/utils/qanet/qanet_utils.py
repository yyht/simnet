import tensorflow as tf
import numpy as np
from model.utils.qanet import qanet_layers

def trilinear_attention(ques_emb, context_emb, ques_mask,
                        context_mask, dropout_keep_prob, config):
    attention_outputs = []
    C = tf.tile(tf.expand_dims(context_emb,2),[1,1,config.max_q_len,1])
    Q = tf.tile(tf.expand_dims(ques_emb,1),[1,config.max_p_len,1,1])
    S = qanet_layers.trilinear([C, Q, C*Q], 
                input_keep_prob = 1.0 - dropout_keep_prob)
    mask_q = tf.expand_dims(ques_mask, 1)
    S_ = tf.nn.softmax(qanet_layers.mask_logits(S, mask = mask_q))
    mask_c = tf.expand_dims(context_mask, 2)
    S_T = tf.transpose(tf.nn.softmax(qanet_layers.mask_logits(S, mask = mask_c), dim = 1),(0,2,1))
    c2q = tf.matmul(S_, ques_emb) # 
    q2c = tf.matmul(tf.matmul(S_, S_T), context_emb)
    attention_outputs.extend([context_emb, c2q, context_emb * c2q])
    if config.q2c:
        attention_outputs.append(context_emb * q2c)
       
    return tf.concat(attention_outputs, axis=-1)

def bilinear_attention(ques_emb, context_emb, ques_mask,
                        context_mask, dropout_keep_prob, config):

    attention_outputs = []

    context_ = tf.transpose(context_emb, [0,2,1])
    hiddem_dim = ques_emb.get_shape()[-1]

    attn_W = tf.get_variable("AttnW",
                        shape=[hiddem_dim, hiddem_dim],
                        dtype=tf.float32,
                        initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                 mode='FAN_AVG',
                                                 uniform=True,
                                                 dtype=tf.float32))

    weighted_query = tf.tensordot(ques_emb, attn_W, axes=[[2], [0]])

    S = tf.matmul(weighted_query, context_)  # batch x q_len x c_len

    mask_q = tf.expand_dims(ques_mask, 1) # batch x 1 x q_len
    mask_c = tf.expand_dims(context_mask, 1) # batch x 1 x c_len

    S_max = tf.nn.softmax(tf.expand_dims(tf.reduce_max(qanet_layers.mask_logits(S, mask = mask_c), 
                                        axis=1), 1), 
                        -1) # batch x 1 x c_len
    c2q = tf.matmul(S_max, context_emb)
    
    S_T = tf.nn.softmax(qanet_layers.mask_logits(tf.transpose(S, [0,2,1]), mask = mask_q)) # batch x c_len x q_len
    q2c = tf.matmul(S_T, ques_emb) # batch x c_len x c_dim
    
    attention_outputs.extend([context_emb, q2c, context_emb * q2c])
    if config.q2c:
        attention_outputs.append(context_emb*c2q)
    return tf.concat(attention_outputs, axis=-1)

def multihead_attention(ques_emb, context_emb, ques_mask,
                        context_mask, 
                        ques_len,
                        context_len,
                        dropout_keep_prob, 
                        is_training,
                        config):

    attention_outputs = qanet_layers.multihead_attention(
                        queries=context_emb,
                        units=config.units,
                        num_heads=config.num_heads,
                        memory=ques_emb,
                        seq_len=context_len,
                        scope="query2context",
                        reuse=None,
                        mask=ques_mask,
                        is_training=is_training,
                        bias=True,
                        dropout=dropout_keep_prob)
    return attention_outputs