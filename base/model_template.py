import tensorflow as tf
import numpy as np
from abc import ABCMeta, abstractmethod
from model.utils.embed import integration_func
import os
from model.utils.representation_model import representation_model_utils 
from model.utils.constraints import weight_constraints

EPSILON = 1e-8

class ModelTemplate(object):
    __metaclass__ = ABCMeta
    def __init__(self):
        self.graph = tf.Graph()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)

        session_conf = tf.ConfigProto(
          allow_soft_placement=True,
          log_device_placement=True,
          gpu_options=gpu_options)
        self.sess = tf.Session(config=session_conf,
                                graph=self.graph)

    def build_placeholder(self, config):
        self.config = config
        with self.graph.as_default():
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
            # input batch size is three times of gold_label

            self.sent1_token = tf.placeholder(tf.int32, [None, None], name='sent1_token')
            self.sent2_token = tf.placeholder(tf.int32, [None, None], name='sent2_token')
            self.gold_label = tf.placeholder(tf.int32, [None], name='gold_label')

            self.start_label = tf.placeholder(tf.int32, [None], name='start_label')
            self.end_label = tf.placeholder(tf.int32, [None], name='end_label')
            
            self.ans_token = tf.placeholder(tf.int32, [None, None], name='answer')
            self.ans_mask = tf.cast(self.ans_token, tf.bool)
            self.ans_token_len = tf.reduce_sum(tf.cast(self.ans_mask, tf.int32), -1)
            
            self.sent1_token_mask = tf.cast(self.sent1_token, tf.bool)
            self.sent1_token_len = tf.reduce_sum(tf.cast(self.sent1_token_mask, tf.int32), -1)
            self.sent2_token_mask = tf.cast(self.sent2_token, tf.bool)
            self.sent2_token_len = tf.reduce_sum(tf.cast(self.sent2_token_mask, tf.int32), -1)

            if self.config.with_char:
                
                self.sent1_char = tf.placeholder(tf.int32, [None, None, None]) # [batch_size, question_len, q_char_len]
                self.sent2_char = tf.placeholder(tf.int32, [None, None, None]) # [batch_size, passage_len, p_char_len]

                self.sent1_char_mask = tf.cast(self.sent1_char, tf.bool)
                self.sent1_char_len = tf.reduce_sum(tf.cast(self.sent1_char_mask, tf.int32), -1)
                self.sent2_char_mask = tf.cast(self.sent2_char, tf.bool)
                self.sent2_char_len = tf.reduce_sum(tf.cast(self.sent2_char_mask, tf.int32), -1)

                self.ans_char = tf.placeholder(tf.int32, [None, None, None]) # [batch_size, answer_len, answer_char_len]
                self.ans_char_mask = tf.cast(self.ans_char, tf.bool)
                self.ans_char_len = tf.reduce_sum(tf.cast(self.ans_char_mask, tf.int32), -1)

                self.char_mat = integration_func.generate_embedding_mat(self.vocab_size, emb_len=self.emb_size,
                                     init_mat=self.token_emb_mat, 
                                     extra_symbol=self.extra_symbol, 
                                     scope=self.scope+'_gene_char_emb_mat')

            self.emb_mat = integration_func.generate_embedding_mat(self.vocab_size, emb_len=self.emb_size,
                                     init_mat=self.token_emb_mat, 
                                     extra_symbol=self.extra_symbol, 
                                     scope=self.scope+'_gene_token_emb_mat')

            if self.config.apply_elmo:
                self.elmo_mat = integration_func.generate_embedding_mat(self.elmo_vocab_size, 
                                     emb_len=self.elmo_emb_size,
                                     init_mat=self.elmo_token_emb_mat, 
                                     extra_symbol=self.extra_symbol, 
                                     scope=self.scope+'_elmo_gene_token_emb_mat')

            # ---------------- for dynamic learning rate -------------------
            self.learning_rate = tf.placeholder(tf.float32, [], 'learning_rate')
            self.dropout_prob = tf.placeholder(tf.float32, name="dropout_prob")
            self.is_training = tf.placeholder(tf.bool, name="is_training")
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            

    @abstractmethod
    def build_embedding(self, *args, **kargs):
        pass

    @abstractmethod
    def build_char_embedding(self, *args, **kargs):
        pass

    @abstractmethod
    def build_model(self, *args, **kargs):
        pass

    @abstractmethod
    def build_encoder(self, *args, **kargs):
        pass

    @abstractmethod
    def build_interactor(self, *args, **kargs):
        pass

    @abstractmethod
    def build_predictor(self, *args, **kargs):
        pass

    @abstractmethod
    def build_loss(self, *args, **kargs):
        pass

    @abstractmethod
    def build_accuracy(self, *args, **kargs):
        pass

    def apply_ema(self, *args, **kargs):
        decay = self.config.get("with_moving_average", None)
        if decay:
            with self.graph.as_default():
                self.var_ema = tf.train.ExponentialMovingAverage(decay)
                ema_op = self.var_ema.apply(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope))
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

    def apply_var_clip(self, *args, **kargs):
        self.clip_vars = []
        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope):
            normalized_var = representation_model_utils.clip_var(var)
            self.clip_vars.append(tf.assign(var, normalized_var))

    def build_op(self, *args, **kargs):

        with self.graph.as_default():
        
            self.build_model(*args, **kargs)
            self.build_loss(*args, **kargs)
            self.build_accuracy(*args, **kargs)

            self.apply_ema(*args, **kargs)

            # ---------- optimization ---------
            if self.config["optimizer"].lower() == 'adadelta':
                self.opt = tf.train.AdadeltaOptimizer(self.learning_rate)
            elif self.config["optimizer"].lower() == 'adam':
                self.opt = tf.train.AdamOptimizer(self.learning_rate)
            elif self.config["optimizer"].lower() == 'rmsprop':
                self.opt = tf.train.RMSPropOptimizer(self.learning_rate)
            elif self.config["optimizer"].lower() == 'adagrad':
                self.opt = tf.train.AdagradOptimizer(self.learning_rate)
                    
            trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
            if self.config.get("weight_constraint", None) == "frobenius":
                self.weight_constraint_loss = self.config.weight_decay*weight_constraints.frobenius_norm(trainable_vars)
                self.loss += self.weight_constraint_loss
            elif self.config.get("weight_constraint", None) == "spectral":
                print("==apply spectral constraints==")
                self.weight_constraint_loss = self.config.weight_decay*weight_constraints.spectral_norm(trainable_vars)
                self.loss += self.weight_constraint_loss

            grads_and_vars = self.opt.compute_gradients(self.loss, var_list=trainable_vars)

            params = [var for _, var in grads_and_vars]
            grads = [grad for grad, _ in grads_and_vars]

            if self.config.get("metric", None) == "Hyperbolic":
                print("====apply weight normalization====")
                self.apply_var_clip()
                grads = [representation_model_utils.H2E_ball(grad, var) \
                                            for grad, var in zip(grads, params)]
            else:
                self.grads, _ = tf.clip_by_global_norm(grads, self.grad_clipper)

            
            self.train_op = self.opt.apply_gradients(zip(grads, params), global_step=self.global_step)
            self.saver = tf.train.Saver(max_to_keep=100)

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
        
    def step(self, batch_samples, *args, **kargs):
        feed_dict = self.get_feed_dict(batch_samples, *args, **kargs)
        with self.graph.as_default():
            [loss, train_op, global_step, 
            accuracy, preds] = self.sess.run([self.loss, self.train_op, 
                                          self.global_step, 
                                          self.accuracy, 
                                          self.pred_probs],
                                          feed_dict=feed_dict)
            # if self.config.metric == "Hyperbolic":
            #     self.sess.run(self.clip_vars)
        return [loss, train_op, global_step, 
                    accuracy, preds]

    def infer(self, batch_samples, mode, *args, **kargs):
        feed_dict = self.get_feed_dict(batch_samples, *args, **kargs)
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

