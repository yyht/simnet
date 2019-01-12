import tensorflow as tf
from model.utils.diin.util import blocks
from model.utils.diin.tensorflow.nn import softsel, get_logits, highway_network, multi_conv1d, linear, conv2d, cosine_similarity, variable_summaries, dense_logits, fuse_gate
from model.utils.diin.tensorflow import flatten, reconstruct, add_wd, exp_mask
import numpy as np

import tensorflow as tf
from model.utils.bimpm import layer_utils, match_utils
from model.utils.qanet import qanet_layers
from model.utils.embed import char_embedding_utils
from loss import point_wise_loss
from base.model_template import ModelTemplate
from model.utils.esim import esim_utils
from model.utils.slstm import slstm_utils
from model.utils.biblosa import cnn, nn, context_fusion, general, rnn, self_attn

from model.utils.diin import diin_utils

EPSILON = 1e-8

class DIIN(ModelTemplate):
    def __init__(self):
        super(DIIN, self).__init__()

    def build_char_embedding(self, char_token, char_lengths, char_embedding, *args, **kargs):

        reuse = kargs["reuse"]
        if self.config.char_embedding == "lstm":
            char_emb = char_embedding_utils.lstm_char_embedding(char_token, char_lengths, char_embedding, 
                            self.config, self.is_training, reuse)
        elif self.config.char_embedding == "conv":
            char_emb = char_embedding_utils.conv_char_embedding(char_token, char_lengths, char_embedding, 
                            self.config, self.is_training, reuse)
        return char_emb

    def build_emebdding(self, index, *args, **kargs):

        reuse = kargs["reuse"]
        dropout_rate = tf.cond(self.is_training, 
                            lambda:self.config.dropout_rate,
                            lambda:0.0)

        if index == "question":
            word_emb = tf.nn.embedding_lookup(self.emb_mat, self.sent1_token)
            if self.config.with_char:
                char_emb = self.build_char_embedding(self.sent1_char, self.sent1_char_len, self.char_mat,
                        is_training=is_training, reuse=reuse)
                word_emb = tf.concat([word_emb, char_emb], axis=-1)
        elif index == "passage":
            word_emb = tf.nn.embedding_lookup(self.emb_mat, self.sent2_token)
            if self.config.with_char:
                char_emb = self.build_char_embedding(self.sent2_char, self.sent2_char_len, self.char_mat,
                        is_training=is_training, reuse=reuse)
                word_emb = tf.concat([word_emb, char_emb], axis=-1)
        return word_emb

    def build_encoder(self, index, input_lengths, *args, **kargs):

        reuse = kargs["reuse"]
        word_emb = self.build_emebdding(index, *args, **kargs)
        dropout_rate = tf.cond(self.is_training, 
                            lambda:self.config.dropout_rate,
                            lambda:0.0)
        input_mask = kargs["input_mask"]
        input_mask = tf.expand_dims(input_mask, -1)

        word_emb = tf.nn.dropout(word_emb, 1-dropout_rate)
        with tf.variable_scope(self.config.scope+"_input_highway", reuse=reuse):
            input_dim = word_emb.get_shape()[-1]
            sent_repres = highway_network(word_emb, self.config.highway_layer_num, True, wd=self.config.wd, is_train=self.is_training)    
            
            for i in range(self.config.self_att_enc_layers):
                with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                    sent_repres = diin_utils.self_attention_layer(self.config, 
                                                    self.is_training,
                                                    sent_repres,
                                                    p_mask=input_mask,
                                                    scope="self_attention_".format(i))

            return sent_repres

    def build_interactor(self, sent1_repres, sent2_repres, sent1_len, sent2_len,
                        sent1_mask, sent2_mask, *args, **kargs):
        reuse = kargs["reuse"]
        input_dim = sent1_repres.get_shape()[-1]
        dropout_rate = tf.cond(self.is_training, 
                            lambda:self.config.dropout_rate,
                            lambda:0.0)

        def model_one_side(config, main, support, 
                            main_length, support_length, 
                            main_mask, support_mask, scope,
                            reuse=None):
            with tf.variable_scope(self.config.scope+"_model_one_side", reuse=reuse):
                bi_att_mx = diin_utils.bi_attention_mx(config, self.is_train, main, support, p_mask=main_mask, h_mask=support_mask) # [N, PL, HL]
               
                bi_att_mx = tf.cond(self.is_training, lambda: tf.nn.dropout(bi_att_mx, 1 - self.config.dropout_rate), lambda: bi_att_mx)
                out_final = diin_utils.dense_net(config, bi_att_mx, self.is_train)
                
                return out_final

        sent1_sent2 = model_one_side(self.config, sent1_repres,
                        sent2_repres,
                        sent1_len,
                        sent2_len,
                        sent1_mask, 
                        sent2_mask,
                        scope=self.config.scope+"interactor",
                        reuse=None)

        sent2_sent1 = model_one_side(self.config, sent2_repres,
                        sent1_repres,
                        sent2_len,
                        sent1_len,
                        sent2_mask, 
                        sent1_mask,
                        scope=self.config.scope+"interactor",
                        reuse=True)

        output = tf.concat([sent1_sent2, sent2_sent1,
                    tf.abs(sent2_sent1 - sent1_sent2),
                    sent1_sent2*sent2_sent1])

        return output

    def build_predictor(self, matched_repres, *args, **kargs):
        reuse = kargs["reuse"]
        num_classes = self.config.num_classes
        dropout_rate = tf.cond(self.is_training, 
                            lambda:self.config.dropout_rate,
                            lambda:0.0)

        self.logits = linear(matched_repres, num_classes ,True, 
                bias_start=0.0, scope=selof.scope+"_logit", 
                squeeze=False, wd=self.config.wd, 
                input_keep_prob=1 - self.config.dropout_rate,
                is_train=self.is_training)
        self.pred_probs = tf.nn.softmax(self.logits)

    def build_loss(self, *args, **kargs):
        if self.config.loss == "softmax_loss":
            self.loss, _ = point_wise_loss.softmax_loss(self.logits, self.gold_label, 
                                    *args, **kargs)
        elif self.config.loss == "sparse_amsoftmax_loss":
            self.loss, _ = point_wise_loss.sparse_amsoftmax_loss(self.logits, self.gold_label, 
                                        self.config, *args, **kargs)
        elif self.config.loss == "focal_loss_binary_v2":
            self.loss, _ = point_wise_loss.focal_loss_binary_v2(self.logits, self.gold_label, 
                                        self.config, *args, **kargs)
        if self.config.l2_loss:
            if self.config.sigmoid_growing_l2loss:
                weights_added = tf.add_n([tf.nn.l2_loss(tensor) for tensor in tf.trainable_variables() if tensor.name.endswith("weights:0") and not tensor.name.endswith("weighted_sum/weights:0") or tensor.name.endswith('kernel:0')])
                full_l2_step = tf.constant(self.config.weight_l2loss_step_full_reg , dtype=tf.int32, shape=[], name='full_l2reg_step')
                full_l2_ratio = tf.constant(self.config.l2_regularization_ratio , dtype=tf.float32, shape=[], name='l2_regularization_ratio')
                gs_flt = tf.cast(self.global_step , tf.float32)
                half_l2_step_flt = tf.cast(full_l2_step / 2 ,tf.float32)
                l2loss_ratio = tf.sigmoid( ((gs_flt - half_l2_step_flt) * 8) / half_l2_step_flt) * full_l2_ratio
            else:
                l2loss = tf.add_n([tf.nn.l2_loss(tensor) for tensor in tf.trainable_variables() if tensor.name.endswith("weights:0") or tensor.name.endswith('kernel:0')]) * tf.constant(config.l2_regularization_ratio , dtype='float', shape=[], name='l2_regularization_ratio')
            
            self.loss += l2loss

    def build_accuracy(self, *args, **kargs):
        self.pred_label = tf.argmax(self.logits, axis=-1)
        print(self.pred_label)
        correct = tf.equal(
            tf.cast(self.pred_label, tf.int32),
            tf.cast(self.gold_label, tf.int32)
        )
        self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    def build_model(self, *args, **kargs):

        self.sent1_encoded = self.build_encoder("question", 
                                        self.sent1_token_len, 
                                        reuse = None,
                                        input_mask=self.sent1_token_mask)

        self.sent2_encoded = self.build_encoder("passage", 
                                        self.sent2_token_len, 
                                        reuse = True,
                                        input_mask=self.sent2_token_mask)

        self.aggregat_repres = self.build_interactor(self.sent1_encoded, 
                                    self.sent2_encoded,
                                    self.sent1_token_len, 
                                    self.sent2_token_len,
                                    self.sent1_token_mask,
                                    self.sent2_token_mask,
                                    reuse = None)

        self.build_predictor(self.aggregat_repres,
                            reuse = None)

    def get_feed_dict(self, sample_batch, *args, **kargs):
        [sent1_token, sent2_token, gold_label] = sample_batch

        feed_dict = {
            self.sent1_token: sent1_token,
            self.sent2_token: sent2_token,
            self.gold_label:gold_label,
            self.learning_rate: self.config.learning_rate,
            self.is_training:kargs["is_training"]
        }
        return feed_dict