import tensorflow as tf
import numpy as np

from metric import pair_wise_metric
import logging

def contrastive_loss(distance, labels, config,
                    *args, **kargs):
    labels = tf.cast(labels, tf.float32)
    distance = tf.squeeze(distance, axis=-1)
    # distance is normalized to [0,1] and the less, the better
    tmp = labels * tf.square(distance)
    tmp2 = (1-labels) *tf.square(tf.maximum((config.loss_margin - distance), 0))
    return tf.reduce_mean(tmp+tmp2) / 2

def hyper_contrastive_loss(distance, labels, config,
                    *args, **kargs):
    labels = tf.cast(labels, tf.float32)
    distance = tf.squeeze(distance, axis=-1)
    # distance is normalized to [0,1] and the less, the better
    tmp = labels * tf.square(distance)
    tmp2 = (1-labels) *tf.square(tf.maximum((config.loss_margin - distance), 0))
    return tf.reduce_mean(tmp+tmp2) / 2






