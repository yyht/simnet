import tensorflow as tf
import numpy as np
from abc import ABCMeta, abstractmethod
from model.utils.embed import integration_func
import os
from model.utils.bimpm import layer_utils, match_utils

from loss import point_wise_loss

import time
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from model.utils.dsmm.utils import os_utils
from model.utils.dsmm.tf_common.optimizer import *
from model.utils.dsmm.tf_common.nn_module import word_dropout, mlp_layer
from model.utils.dsmm.tf_common.nn_module import encode, attend
from model.utils.constraints import weight_constraints

EPSILON = 1e-8

class BaseModel(object):
    __metaclass__ = ABCMeta
    def __init__(self, *args, **kargs):
        self.graph = tf.Graph()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)

        session_conf = tf.ConfigProto(
          allow_soft_placement=True,
          log_device_placement=True,
          gpu_options=gpu_options)
        self.sess = tf.Session(config=session_conf,
                                graph=self.graph)

    def build_placeholder(self, config):
        self.config = config
        with self.graph.as_default():
            self.model_name = self.config.model_name
            self.token_emb_mat = self.config["token_emb_mat"]
            self.char_emb_mat = self.config["char_emb_mat"]
            self.vocab_size = int(self.config["vocab_size"])
            self.char_vocab_size = int(self.config["char_vocab_size"])
            self.max_length = int(self.config["max_length"])
            self.emb_size = int(self.config["emb_size"])
            self.extra_symbol = self.config["extra_symbol"]
            self.scope = self.config["scope"]
            self.num_classes = int(self.config["num_classes"])
            self.batch_size = int(self.config["batch_size"])
            self.grad_clipper = float(self.config.get("grad_clipper", 10.0))
            self.char_limit = self.config.get("char_limit", 10)
            self.char_dim = self.config.get("char_emb_size", 300)

            # ---- add EMLO embedding ----
            if self.config.apply_elmo:
                self.elmo_token_emb_mat = self.config["elmo_token_emb_mat"]
                self.elmo_vocab_size = int(self.config["elmo_vocab_size"])
                self.elmo_emb_size = int(self.config["elmo_emb_size"])
            
            # ---- place holder -----
            self.seq_word_left = tf.placeholder(tf.int32, [None, None], name='seq_word_left')
            self.seq_word_right = tf.placeholder(tf.int32, [None, None], name='seq_word_right')
            self.labels = tf.placeholder(tf.int32, [None], name='gold_label')
            # self.sent1_token_len = tf.placeholder(tf.int32, [None], name='sent1_token_lengths')
            # self.sent2_token_len = tf.placeholder(tf.int32, [None], name='sent2_token_lengths')
            
            self.seq_word_ans = tf.placeholder(tf.int32, [None, None, None], name='seq_word_answer')
            self.seq_len_word_ans = tf.placeholder(tf.int32, shape=[None, None], name="seq_word_len_answer")
            self.seq_word_ans_mask = tf.cast(self.seq_word_ans, tf.bool)
            self.seq_word_ans_len = tf.reduce_sum(tf.cast(self.seq_word_ans_mask, tf.int32), -1)

            self.seq_len_word_left = tf.placeholder(tf.int32, shape=[None], name="seq_len_word_left")
            self.seq_len_word_right = tf.placeholder(tf.int32, shape=[None], name="seq_len_word_right")

            self.seq_word_left_mask = tf.cast(self.seq_word_left, tf.bool)
            self.seq_word_left_len = tf.reduce_sum(tf.cast(self.seq_word_left_mask, tf.int32), -1)
            self.seq_word_right_mask = tf.cast(self.seq_word_right, tf.bool)
            self.seq_word_right_len = tf.reduce_sum(tf.cast(self.seq_word_right_mask, tf.int32), -1)

            if self.config.with_char:
                # self.sent1_char_len = tf.placeholder(tf.int32, [None,None]) # [batch_size, question_len]
                # self.sent2_char_len = tf.placeholder(tf.int32, [None,None]) # [batch_size, passage_len]
                self.seq_char_left = tf.placeholder(tf.int32, [None, None, None], name="seq_char_left") # [batch_size, question_len, q_char_len]
                self.seq_char_right = tf.placeholder(tf.int32, [None, None, None], name="seq_char_right") # [batch_size, passage_len, p_char_len]

                self.seq_len_char_left = tf.placeholder(tf.int32, shape=[None], name="seq_len_char_left")
                self.seq_len_char_right = tf.placeholder(tf.int32, shape=[None], name="seq_len_char_right")

                self.seq_char_left_mask = tf.cast(self.seq_char_left, tf.bool)
                self.seq_char_left_len = tf.reduce_sum(tf.cast(self.seq_char_left_mask, tf.int32), -1)
                self.seq_char_right_mask = tf.cast(self.seq_char_right, tf.bool)
                self.seq_char_right_len = tf.reduce_sum(tf.cast(self.seq_char_right_mask, tf.int32), -1)

                self.seq_char_ans = tf.placeholder(tf.int32, [None, None, None, None], name="seq_char_answer")
                self.seq_char_ans_mask = tf.cast(self.seq_char_ans, tf.bool)
                self.seq_char_ans_len = tf.reduce_sum(tf.cast(self.seq_char_ans_mask, tf.int32), -1)

                self.char_mat = integration_func.generate_embedding_mat(self.vocab_size, emb_len=self.emb_size,
                                     init_mat=self.token_emb_mat, 
                                     extra_symbol=self.extra_symbol, 
                                     scope='gene_char_emb_mat')

            self.emb_mat = integration_func.generate_embedding_mat(self.vocab_size, emb_len=self.emb_size,
                                     init_mat=self.token_emb_mat, 
                                     extra_symbol=self.extra_symbol, 
                                     scope='gene_token_emb_mat')

            if self.config.apply_elmo:
                self.elmo_mat = integration_func.generate_embedding_mat(self.elmo_vocab_size, 
                                     emb_len=self.elmo_emb_size,
                                     init_mat=self.elmo_token_emb_mat, 
                                     extra_symbol=self.extra_symbol, 
                                     scope='elmo_gene_token_emb_mat')

            # ---------------- for dynamic learning rate -------------------
            # self.learning_rate = tf.placeholder(tf.float32, [], 'learning_rate')
            self.dropout_prob = tf.placeholder(tf.float32, name="dropout_prob")
            self.is_training = tf.placeholder(tf.bool, name="is_training")
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            
            #### features
            if self.config.use_features:
                self.features = tf.placeholder(tf.float32, shape=[None, self.config["num_features"]], name="features")
    
            self.learning_rate = tf.train.exponential_decay(self.config["init_lr"], self.global_step,
                                                        self.config["decay_steps"], self.config["decay_rate"])
            self.augmentation_dropout = tf.train.exponential_decay(self.config["augmentation_init_dropout"], self.global_step,
                                                               self.config["augmentation_dropout_decay_steps"],
                                                               self.config["augmentation_dropout_decay_rate"])
            self.augmentation_permutation = tf.train.exponential_decay(self.config["augmentation_init_permutation"],
                                                               self.global_step,
                                                               self.config["augmentation_permutation_decay_steps"],
                                                               self.config["augmentation_permutation_decay_rate"])

    def apply_ema(self, *args, **kargs):
        decay = self.config.get("with_moving_average", None)
        if decay:
            with self.graph.as_default():
                self.var_ema = tf.train.ExponentialMovingAverage(decay)
                ema_op = self.var_ema.apply(tf.trainable_variables())
                with tf.control_dependencies([ema_op]):
                    self.loss = tf.identity(self.loss)

                    self.shadow_vars = []
                    self.global_vars = []
                    for var in tf.global_variables():
                        v = self.var_ema.average(var)
                        if v:
                            self.shadow_vars.append(v)
                            self.global_vars.append(var)
                    self.assign_vars = []
                    for g,v in zip(self.global_vars, self.shadow_vars):
                        self.assign_vars.append(tf.assign(g,v))

    def init_step(self):
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())

    def save_model(self, model_dir, model_str):
        with self.graph.as_default():
            self.saver.save(self.sess, 
                    os.path.join(model_dir, 
                                model_str+".ckpt"))

    def load_model(self, model_dir, model_str):
        with self.graph.as_default():
            model_path = os.path.join(model_dir, model_str+".ckpt")
            self.saver.restore(self.sess, model_path)
            print("======model str======", model_str)
            if self.config.get("with_moving_average", None):
                self.sess.run(self.assign_vars)

    def step(self, batch_samples, is_training, symmetric, 
                *args, **kargs):
        feed_dict = self._get_feed_dict(batch_samples, 
                                    is_training, 
                                    symmetric)
        with self.graph.as_default():
            [loss, train_op, global_step, 
            accuracy, preds] = self.sess.run([self.loss, self.train_op, 
                                          self.global_step, 
                                          self.accuracy, 
                                          self.pred_probs],
                                          feed_dict=feed_dict)
            
        return [loss, train_op, global_step, 
                    accuracy, preds]

    def infer(self, batch_samples, mode, is_training, symmetric, 
                *args, **kargs):
        feed_dict = self._get_feed_dict(batch_samples, 
                                    is_training, 
                                    symmetric)
        if mode == "test":
            with self.graph.as_default():
                [loss, logits, pred_probs, accuracy] = self.sess.run([self.loss, self.logits, 
                                                            self.pred_probs, 
                                                            self.accuracy], 
                                                            feed_dict=feed_dict)
            return loss, logits, pred_probs, accuracy
        elif mode == "infer":
            with self.graph.as_default():
                [logits, pred_probs] = self.sess.run([self.logits, self.pred_probs], 
                                            feed_dict=feed_dict)
            return logits, pred_probs


    def _semantic_feature_layer(self, seq_input, seq_len, granularity="word", reuse=False):
        assert granularity in ["char", "word"]
        with self.graph.as_default():
            emb_seq = tf.nn.embedding_lookup(self.emb_mat, seq_input)
            dropout_rate = tf.cond(self.is_training, 
                                lambda:self.config.dropout_rate,
                                lambda:0.0)

            input_dim = emb_seq.get_shape()[-1]
            with tf.variable_scope(self.config.scope+"_input_highway", reuse=reuse): 
                emb_seq = match_utils.multi_highway_layer(emb_seq, 
                        input_dim, self.config.highway_layer_num)

            emb_seq = tf.nn.dropout(emb_seq, 1 - dropout_rate)

            #### encode
            input_dim = self.emb_size  #self.config["embedding_dim"]
            enc_seq = encode(emb_seq, method=self.config["encode_method"],
                             input_dim=input_dim,
                             params=self.config,
                             sequence_length=seq_len,
                             mask_zero=self.config["embedding_mask_zero"],
                             scope_name=self.scope + "enc_seq_%s"%granularity, 
                             reuse=reuse,
                             training=self.is_training)

            #### attend
            feature_dim = self.config["encode_dim"]
            print("==semantic feature dim==", feature_dim, enc_seq.get_shape())
            context = None

            att_seq = attend(enc_seq, context=context,
                             encode_dim=self.config["encode_dim"],
                             feature_dim=feature_dim,
                             attention_dim=self.config["attention_dim"],
                             method=self.config["attend_method"],
                             scope_name=self.scope + "att_seq_%s"%granularity,
                             reuse=reuse, num_heads=self.config["attention_num_heads"])
            print("==semantic layer attention seq shape==", att_seq.get_shape())
            #### MLP nonlinear projection
            sem_seq = mlp_layer(att_seq, fc_type=self.config["fc_type"],
                                hidden_units=self.config["fc_hidden_units"],
                                dropouts=self.config["fc_dropouts"],
                                scope_name=self.scope + "sem_seq_%s"%granularity,
                                reuse=reuse,
                                training=self.is_training,
                                seed=self.config["random_seed"])
            print("==semantic layer mlp seq shape==", sem_seq.get_shape())
            return emb_seq, enc_seq, att_seq, sem_seq

    def _interaction_semantic_feature_layer(self, seq_input_left, 
            seq_input_right, seq_len_left, seq_len_right, 
            granularity="word"):
        assert granularity in ["char", "word"]
        #### embed
        with self.graph.as_default():
            emb_seq_left = tf.nn.embedding_lookup(self.emb_mat, seq_input_left)
            dropout_rate = tf.cond(self.is_training, 
                                lambda:self.config.dropout_rate,
                                lambda:0.0)

            input_dim = emb_seq_left.get_shape()[-1]
            with tf.variable_scope(self.config.scope+"_input_highway", reuse=False): 
                emb_seq_left = match_utils.multi_highway_layer(emb_seq_left, 
                        input_dim, self.config.highway_layer_num)

            emb_seq_left = tf.nn.dropout(emb_seq_left, 1 - dropout_rate)

            seq_input_right = tf.nn.embedding_lookup(self.emb_mat, seq_input_right)
            dropout_rate = tf.cond(self.is_training, 
                                lambda:self.config.dropout_rate,
                                lambda:0.0)

            input_dim = seq_input_right.get_shape()[-1]
            with tf.variable_scope(self.config.scope+"_input_highway", reuse=True): 
                seq_input_right = match_utils.multi_highway_layer(seq_input_right, 
                        input_dim, self.config.highway_layer_num)

            seq_input_right = tf.nn.dropout(seq_input_right, 1 - dropout_rate)

            #### encode
            input_dim = self.emb_size #self.config["embedding_dim"]
            enc_seq_left = encode(emb_seq_left, method=self.config["encode_method"],
                                  input_dim=input_dim,
                                  params=self.config,
                                  sequence_length=seq_len_left,
                                  mask_zero=self.config["embedding_mask_zero"],
                                  scope_name=self.scope + "enc_seq_%s"%granularity, reuse=False,
                                  training=self.is_training)
            enc_seq_right = encode(emb_seq_right, method=self.config["encode_method"],
                                   input_dim=input_dim,
                                   params=self.config,
                                   sequence_length=seq_len_right,
                                   mask_zero=self.config["embedding_mask_zero"],
                                   scope_name=self.scope + "enc_seq_%s" % granularity, reuse=True,
                                   training=self.is_training)

            #### attend
            # [batchsize, s1, s2]
            att_mat = tf.einsum("abd,acd->abc", enc_seq_left, enc_seq_right)
            feature_dim = self.config["encode_dim"] + self.config["max_seq_len_%s"%granularity]
            att_seq_left = attend(enc_seq_left, context=att_mat, feature_dim=feature_dim,
                                       method=self.config["attend_method"],
                                       scope_name=self.scope + "att_seq_%s"%granularity,
                                       reuse=False)
            att_seq_right = attend(enc_seq_right, context=tf.transpose(att_mat), feature_dim=feature_dim,
                                  method=self.config["attend_method"],
                                  scope_name=self.scope + "att_seq_%s" % granularity,
                                  reuse=True)

            #### MLP nonlinear projection
            sem_seq_left = mlp_layer(att_seq_left, fc_type=self.config["fc_type"],
                                     hidden_units=self.config["fc_hidden_units"],
                                     dropouts=self.config["fc_dropouts"],
                                     scope_name=self.scope + "sem_seq_%s"%granularity,
                                     reuse=False,
                                     training=self.is_training,
                                     seed=self.config["random_seed"])
            sem_seq_right = mlp_layer(att_seq_right, fc_type=self.config["fc_type"],
                                      hidden_units=self.config["fc_hidden_units"],
                                      dropouts=self.config["fc_dropouts"],
                                      scope_name=self.scope + "sem_seq_%s" % granularity,
                                      reuse=True,
                                      training=self.is_training,
                                      seed=self.config["random_seed"])

            return emb_seq_left, enc_seq_left, att_seq_left, sem_seq_left, \
                    emb_seq_right, enc_seq_right, att_seq_right, sem_seq_right

    def _get_matching_features(self):
        pass

    def _get_prediction(self):
        with self.graph.as_default():
            with tf.name_scope(self.model_name + "/"):
                with tf.name_scope("prediction"):
                    lst = []
                    if "word" in self.config["granularity"]:
                        lst.append(self.matching_features_word)
                    # if "char" in self.config["granularity"]:
                    #     lst.append(self.matching_features_char)
                    if self.config["use_features"]:
                        out_0 = mlp_layer(self.features, fc_type=self.config["fc_type"],
                                          hidden_units=self.config["fc_hidden_units"],
                                          dropouts=self.config["fc_dropouts"],
                                          scope_name=self.scope + "mlp_features",
                                          reuse=False,
                                          training=self.is_training,
                                          seed=self.config["random_seed"])
                        lst.append(out_0)
                    out = tf.concat(lst, axis=-1)
                    out = tf.layers.Dropout(self.config["final_dropout"])(out, training=self.is_training)
                    out = mlp_layer(out, fc_type=self.config["fc_type"],
                                    hidden_units=self.config["fc_hidden_units"],
                                    dropouts=self.config["fc_dropouts"],
                                    scope_name=self.scope + "mlp",
                                    reuse=False,
                                    training=self.is_training,
                                    seed=self.config["random_seed"])
                    logits = tf.layers.dense(out, self.num_classes, 
                                            activation=None,
                                             kernel_initializer=tf.glorot_uniform_initializer(
                                             seed=self.config["random_seed"]),
                                             name=self.scope + "logits")

                    proba = tf.nn.softmax(logits)

            return logits, proba

    def _get_accuracy(self, *args, **kargs):
        self.pred_label = tf.argmax(self.logits, axis=-1)
        print(self.pred_label)
        correct = tf.equal(
            tf.cast(self.pred_label, tf.int32),
            tf.cast(self.labels, tf.int32)
        )
        self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))


    def _get_loss(self, *args, **kargs):
        if self.config.loss == "softmax_loss":
            self.loss, _ = point_wise_loss.softmax_loss(self.logits, self.labels, 
                                    *args, **kargs)
        elif self.config.loss == "sparse_amsoftmax_loss":
            self.loss, _ = point_wise_loss.sparse_amsoftmax_loss(self.logits, self.labels, 
                                        self.config, *args, **kargs)
        elif self.config.loss == "focal_loss_binary_v2":
            self.loss, _ = point_wise_loss.focal_loss_binary_v2(self.logits, self.labels, 
                                        self.config, *args, **kargs)
        elif self.config.loss == "focal_loss_multi_v1":
            self.loss, _ = point_wise_loss.focal_loss_multi_v1(self.logits, self.labels, 
                                        self.config, *args, **kargs)

    def build_op(self, *args, **kargs):
        with self.graph.as_default():
            self.matching_features_word = self._get_matching_features()
            print(self.matching_features_word.get_shape(), "===matching_features_word_dim===")
            self.logits, self.pred_probs = self._get_prediction()
            print(self.logits.get_shape(), self.pred_probs.get_shape(), "===")
            self._get_loss()
            print(self.loss.get_shape(), "===loss shape===")
            self.train_op = self._get_train_op()
            self._get_accuracy()
            self.apply_ema(*args, **kargs)
            self.saver = tf.train.Saver(max_to_keep=100)

    def _get_train_op(self):
        with tf.name_scope(self.config.model_name + "/"):
            with tf.name_scope("optimization"):
                if self.config["optimizer_type"] == "lazynadam":
                    optimizer = LazyNadamOptimizer(learning_rate=self.learning_rate, beta1=self.config["beta1"],
                                                 beta2=self.config["beta2"], epsilon=1e-8,
                                                 schedule_decay=self.config["schedule_decay"])
                elif self.config["optimizer_type"] == "adam":
                    optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                     beta1=self.config["beta1"],
                                                     beta2=self.config["beta2"], epsilon=1e-8)
                elif self.config["optimizer_type"] == "lazyadam":
                    optimizer = tf.contrib.opt.LazyAdamOptimizer(learning_rate=self.learning_rate,
                                                               beta1=self.config["beta1"],
                                                               beta2=self.config["beta2"], epsilon=1e-8)
                elif self.config["optimizer_type"] == "adagrad":
                    optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                            initial_accumulator_value=1e-7)
                elif self.config["optimizer_type"] == "adadelta":
                    optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate)
                elif self.config["optimizer_type"] == "gd":
                      optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
                elif self.config["optimizer_type"] == "momentum":
                    optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95)
                elif self.config["optimizer_type"] == "rmsprop":
                    optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, decay=0.9,
                                                            momentum=0.9, epsilon=1e-8)
                elif self.config["optimizer_type"] == "lazypowersign":
                    optimizer = LazyPowerSignOptimizer(learning_rate=self.learning_rate)
                elif self.config["optimizer_type"] == "lazyaddsign":
                    optimizer = LazyAddSignOptimizer(learning_rate=self.learning_rate)
                elif self.config["optimizer_type"] == "lazyamsgrad":
                    optimizer = LazyAMSGradOptimizer(learning_rate=self.learning_rate, beta1=self.config["beta1"],
                                                       beta2=self.config["beta2"], epsilon=1e-8)

                trainable_vars = tf.trainable_variables()
                if self.config.get("weight_constraint", None) == "frobenius":
                    self.weight_constraint_loss = self.config.weight_decay*weight_constraints.frobenius_norm(trainable_vars)
                    self.loss += self.weight_constraint_loss
                elif self.config.get("weight_constraint", None) == "spectral":
                    self.weight_constraint_loss = self.config.weight_decay*weight_constraints.spectral_norm(trainable_vars)
                    self.loss += self.weight_constraint_loss

                grads_and_vars = optimizer.compute_gradients(self.loss, var_list=trainable_vars)

                params = [var for _, var in grads_and_vars]
                grads = [grad for grad, _ in grads_and_vars]

                grads, _ = tf.clip_by_global_norm(grads, self.grad_clipper)

                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    train_op = optimizer.apply_gradients(zip(grads, params), global_step=self.global_step)
        return train_op

    def _dropout(self, val_arr, ind_arr, p, value):
        new_arr = np.array(val_arr)
        drop = np.empty(val_arr.shape, dtype=np.bool)
        for i in range(val_arr.shape[0]):
            drop[i, :ind_arr[i]] = np.random.choice([True, False], ind_arr[i], p=[p, 1 - p])
        new_arr[drop] = value
        return new_arr

    def _dropout_augmentation(self, feed_dict):

        p = self.sess.run(self.augmentation_dropout)
        if p <= self.config["augmentation_min_dropout"]:
            return

        dropout_data = self._dropout(val_arr=feed_dict[self.seq_word_left],
                                     ind_arr=feed_dict[self.seq_len_word_left],
                                     p=p, value=self.config.MISSING_INDEX_WORD)
        feed_dict[self.seq_word_left] = np.vstack([
            feed_dict[self.seq_word_left],
            dropout_data,
        ])

        dropout_data = self._dropout(val_arr=feed_dict[self.seq_word_right],
                                     ind_arr=feed_dict[self.seq_len_word_right],
                                     p=p, value=self.config.MISSING_INDEX_WORD)
        feed_dict[self.seq_word_right] = np.vstack([
            feed_dict[self.seq_word_right],
            dropout_data,
        ])

        # dropout_data = self._dropout(val_arr=feed_dict[self.seq_char_left],
        #                              ind_arr=feed_dict[self.seq_len_char_left],
        #                              p=p, value=config.MISSING_INDEX_CHAR)
        # feed_dict[self.seq_char_left] = np.vstack([
        #     feed_dict[self.seq_char_left],
        #     dropout_data,
        # ])

        # dropout_data = self._dropout(val_arr=feed_dict[self.seq_char_right],
        #                              ind_arr=feed_dict[self.seq_len_char_right],
        #                              p=p, value=config.MISSING_INDEX_CHAR)
        # feed_dict[self.seq_char_right] = np.vstack([
        #     feed_dict[self.seq_char_right],
        #     dropout_data,
        # ])

        # double others
        feed_dict[self.seq_len_word_left] = np.tile(feed_dict[self.seq_len_word_left], 2)
        feed_dict[self.seq_len_word_right] = np.tile(feed_dict[self.seq_len_word_right], 2)
        # feed_dict[self.seq_len_char_left] = np.tile(feed_dict[self.seq_len_char_left], 2)
        # feed_dict[self.seq_len_char_right] = np.tile(feed_dict[self.seq_len_char_right], 2)
        feed_dict[self.labels] = np.tile(feed_dict[self.labels], 2)
        if self.config["use_features"]:
            feed_dict[self.features] = np.tile(feed_dict[self.features], [2, 1])


    def _permutation(self, val_arr, ind_arr, p):
        if np.random.random() < p:
            new_arr = np.array(val_arr)
            for i in range(val_arr.shape[0]):
                new_arr[i, :ind_arr[i]] = np.random.permutation(new_arr[i,:ind_arr[i]])
            return new_arr
        else:
            return val_arr

    def _permutation_augmentation(self, feed_dict):
        p = self.sess.run(self.augmentation_permutation)
        if p <= self.config["augmentation_min_permutation"]:
            return

        feed_dict[self.seq_word_left] = np.vstack([
            feed_dict[self.seq_word_left],
            self._permutation(feed_dict[self.seq_word_left], feed_dict[self.seq_len_word_left], p),
        ])
        feed_dict[self.seq_word_right] = np.vstack([
            feed_dict[self.seq_word_right],
            self._permutation(feed_dict[self.seq_word_right], feed_dict[self.seq_len_word_right], p),
        ])
        # feed_dict[self.seq_char_left] = np.vstack([
        #     feed_dict[self.seq_char_left],
        #     self._permutation(feed_dict[self.seq_char_left], feed_dict[self.seq_len_char_left], p),
        # ])
        # feed_dict[self.seq_char_right] = np.vstack([
        #     feed_dict[self.seq_char_right],
        #     self._permutation(feed_dict[self.seq_char_right], feed_dict[self.seq_len_char_right], p),
        # ])
        # double others
        feed_dict[self.seq_len_word_left] = np.tile(feed_dict[self.seq_len_word_left], 2)
        feed_dict[self.seq_len_word_right] = np.tile(feed_dict[self.seq_len_word_right], 2)
        # feed_dict[self.seq_len_char_left] = np.tile(feed_dict[self.seq_len_char_left], 2)
        # feed_dict[self.seq_len_char_right] = np.tile(feed_dict[self.seq_len_char_right], 2)
        feed_dict[self.labels] = np.tile(feed_dict[self.labels], 2)
        if self.config["use_features"]:
            feed_dict[self.features] = np.tile(feed_dict[self.features], [2, 1])

    def _get_feed_dict(self, Q, training=False, symmetric=False):
        if training:
            q1 = Q["words"][0]
            q2 = Q["words"][1]
            q1_len = Q["seq_len_words"][0]
            q2_len = Q["seq_len_words"][1]
            label = Q["labels"]

            feed_dict = {
                self.seq_word_left:  q1,
                self.seq_word_right: q2,
                self.seq_len_word_left:  q1_len,
                self.seq_len_word_right: q2_len,
                self.labels: label,
                self.is_training: training,
            }
            # if self.config.with_char:
            #     q1_char = Q["chars"][0]
            #     q2_char = Q["chars"][1]
            #     q1_char_len = Q["seq_len_chars"][0]
            #     q2_char_len = Q["seq_len_chars"][1]
            #     feed_dict[self.seq_char_left] = q1_char
            #     feed_dict[self.seq_char_right] = q2_char
            #     feed_dict[self.seq_char_left_len] = q1_char_len
            #     feed_dict[self.seq_char_right_len] = q2_char_len

            if self.config["use_features"]:
                feed_dict.update({
                    self.features: Q["features"]
                })
        elif not symmetric:
            q1 = Q["words"][0]
            q2 = Q["words"][1]
            q1_len = Q["seq_len_words"][0]
            q2_len = Q["seq_len_words"][1]
            label = Q["labels"]

            feed_dict = {
                self.seq_word_left: q1,
                self.seq_word_right: q2,
                self.seq_len_word_left: q1_len,
                self.seq_len_word_right: q2_len,
                self.labels: label,
                self.is_training: training,
            }
            if self.config["use_features"]:
                feed_dict.update({
                    self.features: Q["features"],
                })
            # if self.config.with_char:
            #     q1_char = Q["chars"][0]
            #     q2_char = Q["chars"][1]
            #     q1_char_len = Q["seq_len_chars"][0]
            #     q2_char_len = Q["seq_len_chars"][1]
            #     feed_dict[self.seq_char_left] = q1_char
            #     feed_dict[self.seq_char_right] = q2_char
            #     feed_dict[self.seq_char_left_len] = q1_char_len
            #     feed_dict[self.seq_char_right_len] = q2_char_len

        else:
            q1 = Q["words"][0]
            q2 = Q["words"][1]
            q1_len = Q["seq_len_words"][0]
            q2_len = Q["seq_len_words"][1]
            label = Q["labels"]

            feed_dict = {
                self.seq_word_left:  q1,
                self.seq_word_right: q2,
                self.seq_len_word_left:  q1_len,
                self.seq_len_word_right: q2_len,
                self.labels: label,
                self.is_training: training,
            }
            # if self.config.with_char:
            #     q1_char = Q["chars"][0]
            #     q2_char = Q["chars"][1]
            #     q1_char_len = Q["seq_len_chars"][0]
            #     q2_char_len = Q["seq_len_chars"][1]
            #     feed_dict[self.seq_char_left] = q1_char
            #     feed_dict[self.seq_char_right] = q2_char
            #     feed_dict[self.seq_char_left_len] = q1_char_len
            #     feed_dict[self.seq_char_right_len] = q2_char_len

            if self.config["use_features"]:
                feed_dict.update({
                    self.features: Q["features"]
                })
            
        # augmentation
        if training:
            if self.config["augmentation_init_dropout"] > 0:
                with self.graph.as_default():
                    self._dropout_augmentation(feed_dict)
            if self.config["augmentation_init_permutation"] > 0:
                with self.graph.as_default():
                    self._permutation_augmentation(feed_dict)

        return feed_dict
