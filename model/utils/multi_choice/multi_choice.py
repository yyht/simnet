import numpy as np
import tensorflow as tf

def scalar_attention(query, context, query_mask, conetxt_mask,
                    mask_zero=False, scope_name="attention", 
                    reuse=False):
    
    with tf.variable_scope(scope_name, reuse=reuse):

        query_dim = query.get_shape()[-1]
        conetxt_dim = context.get_shape()[-1]

        W1 = tf.get_variable("W1_%s" % scope_name,
                             initializer=tf.truncated_normal_initializer(
                                 mean=0.0, stddev=0.2, dtype=tf.float32, seed=seed),
                             dtype=tf.float32,
                             shape=[query_dim, conetxt_dim])

        b1 = tf.get_variable("b1_%s" % scope_name,
                             initializer=tf.truncated_normal_initializer(
                                 mean=0.0, stddev=0.2, dtype=tf.float32, seed=seed),
                             dtype=tf.float32,
                             shape=[1])

        query_shapes = query.shape.as_list()
        if len(query_shapes) > 4:
            raise NotImplementedError
        elif len(query_shapes) == 4:
        	S = tf.einsum("abcd,de->abce", query, W1)
        	S = tf.einsum("abce,ade->abcd", S, context)

        	S = tf.reduce_max(S, axis=(-1,-2)) # batch x num_choice


        else:
        	S = tf.einsum("abc,cd->abd", query, W1)
        	S = tf.einsum("abd,acd->abc", S, context)



    e = tf.einsum("", x, W1) + \
        tf.expand_dims(b1, axis=1)
    a = tf.exp(e)

    # apply mask after the exp. will be re-normalized next
    if mask_zero:
        # None * s
        mask = tf.sequence_mask(sequence_length, maxlen)
        mask = tf.cast(mask, tf.float32)
        a = a * mask

    # in some cases especially in the early stages of training the sum may be almost zero
    s = tf.reduce_sum(a, axis=1, keep_dims=True)
    a /= tf.cast(s + epsilon, tf.float32)
    a = tf.expand_dims(a, axis=-1)

    return a
