import tensorflow as tf
import numpy as np

def frobenius_norm(var_list):
	frobenius = 0
	for var in var_list:
		shape = var.get_shape().as_list()
		if len(shape) >= 2:
			frobenius += tf.nn.l2_loss(var)
	return 	frobenius

def spectral_norm_(var, itera=1): 
	spectral = 0
	w_shape = var.get_shape().as_list()
	w = tf.reshape(var, [-1, tf.shape(var)[-1]]) # in_dim x output_dim

	u = tf.cast(tf.ones_like(w)[:,0], tf.float32) # in_dim
	u = tf.expand_dims(u, 0) # 1 x in_dim
	for i in range(itera):
		v = tf.nn.l2_normalize(tf.matmul(u, w)) # 1 x output_dim
		u = tf.nn.l2_normalize(tf.matmul(v, tf.transpose(w))) # 1 x in_dim
	spectral = tf.reduce_sum(tf.matmul(tf.matmul(u, w), tf.transpose(v)))
	return spectral

def spectral_norm(var_list, itera=5):
	spectral = 0
	for var in var_list:
		shape = var.get_shape().as_list()
		if len(shape) >= 2:
			spectral += spectral_norm_(var, itera)
	return spectral