from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from model.utils.bimpm import layer_utils, match_utils


def hyperbolic_ball(x, y, eps=1e-8):
    """ Poincare Distance Function.
    """
    z = x - y
    z = tf.norm(z, ord='euclidean', keep_dims=True, axis=1)
    z = tf.square(z)
    x_d = 1 - tf.square(tf.norm(x, ord='euclidean',
                                    keep_dims=True, 
                                    axis=1))
    y_d = 1 - tf.square(tf.norm(y, ord='euclidean',
                                    keep_dims=True, 
                                    axis=1))

    x_d = tf.maximum(x_d, eps)
    y_d = tf.maximum(y_d, eps)

    d = x_d * y_d
    z = z / d
    z  = (2 * z) + 1

    z = tf.maximum(z, 1+eps)

    arcosh = tf.acosh(z)
    arcosh = tf.squeeze(arcosh, axis=1)
    return arcosh

def clip_var(var, eps=1E-5):
    shape = var.get_shape().as_list()

    if len(shape) >= 3:
        var = tf.clip_by_norm(var, 1.-eps, axes=[-2,-1])
    elif len(shape) == 2:
        var = tf.clip_by_norm(var, 1.-eps, axes=1)
    else:
        var = tf.clip_by_norm(var, 1.-eps)

    return var

def distance_transformation(distance, config, reuse=None):
    if config.dist_transformation in ["linear", "sigmoid"]:
        with tf.variable_scope(config.scope+'_fl',reuse=reuse) as scope:
            distance = tf.expand_dims(distance, 1)
            weights_linear = tf.get_variable('final_weights',
                                [1,1],
                                initializer=tf.contrib.layers.xavier_initializer())
            bias_linear = tf.get_variable('bias', 
                                [1],
                                initializer=tf.zeros_initializer())

            final_layer = tf.nn.xw_plus_b(distance, weights_linear, 
                                        bias_linear)
        if config.dist_transformation == "sigmoid":
            final_layer = tf.squeeze(tf.nn.sigmoid(final_layer), axis=1)
        else:
            final_layer = tf.squeeze(final_layer, axis=1)
    elif config.dist_transformation == "exp":
        final_layer = tf.exp(-distance)
    else:
        final_layer = distance
    return final_layer
    
def H2E_ball(grad, var, eps=1E-5):
    ''' Converts hyperbolic gradient to euclidean gradient
    '''
    shape = grad.get_shape().as_list()

    if len(shape) >= 3:
        grad_scale = 1.- tf.square(tf.norm(var, ord='euclidean', axis=[-2, -1], keep_dims=True))
    elif len(shape) == 2:
        grad_scale = 1.- tf.square(tf.norm(var, ord='euclidean', axis=1, keep_dims=True))
    else:
        grad_scale = 1.- tf.square(tf.norm(var, ord='euclidean', keep_dims=True))
    
    grad_scale = tf.square(grad_scale)
    grad_scale = (grad_scale) / 4
    grad = grad * grad_scale
    return grad

def pair_exp_probs(distance):
    """
    Given a pair of encoded sentences (vectors), return a probability
    distribution on whether they are duplicates are not with:
    exp(-||sentence_one - sentence_two||)

    Parameters
    ----------
    sentence_one: Tensor
        A tensor of shape (batch_size, 2*rnn_hidden_size) representing
        the encoded sentence_ones to use in the probability calculation.

    sentence_one: Tensor
        A tensor of shape (batch_size, 2*rnn_hidden_size) representing
        the encoded sentence_twos to use in the probability calculation.

    Returns
    -------
        class_probabilities: Tensor
            A tensor of shape (batch_size, 2), represnting the probability
            that a pair of sentences are duplicates as
            [is_not_duplicate, is_duplicate].
    """
    with tf.name_scope("l1_similarity"):
        # Take the L1 norm of the two vectors.
        # Shape: (batch_size, 2*rnn_hidden_size)
        
        # Take the sum for each sentence pair
        # Shape: (batch_size, 1)

        # Exponentiate the negative summed L1 distance to get the
        # positive-class probability.
        # Shape: (batch_size, 1)
        distance = tf.expand_dims(distance, axis=1) # (batch_size, 1)
        positive_class_probs = tf.exp(-distance)

        # Get the negative class probabilities by subtracting
        # the positive class probabilities from 1.
        # Shape: (batch_size, 1)
        negative_class_probs = 1 - positive_class_probs

        # Concatenate the positive and negative class probabilities
        # Shape: (batch_size, 2)
        class_probabilities = tf.concat([negative_class_probs,
                                             positive_class_probs], 1)

        # if class_probabilities has 0's, then taking the log of it
        # (e.g. for cross-entropy loss) will cause NaNs. So we add
        # epsilon and renormalize by the sum of the vector.
        safe_class_probabilities = class_probabilities + 1e-08
        safe_class_probabilities /= tf.reduce_sum(safe_class_probabilities,
                                                      axis=1,
                                                      keep_dims=True)
        return safe_class_probabilities