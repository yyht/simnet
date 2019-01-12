import tensorflow as tf
from model.utils.bimpm import layer_utils, match_utils
from model.utils.qanet import qanet_layers
from model.utils.embed import char_embedding_utils
from loss import point_wise_loss
from base.model_template import ModelTemplate
from model.utils.esim import esim_utils
from model.utils.slstm import slstm_utils
from model.utils.biblosa import cnn, nn, context_fusion, general, rnn, self_attn
from model.utils.hard_attention import hard_attention_utils
import numpy as np
EPSILON = 1e-8

class HardAttention(ModelTemplate):
    def __init__(self):
        super(HardAttention, self).__init__()

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
            if self.config.apply_elmo:
                elmo_emb = tf.nn.embedding_lookup(self.elmo_mat, self.sent1_token)
                word_emb = tf.concat([word_emb, elmo_emb], axis=-1)
        elif index == "passage":
            word_emb = tf.nn.embedding_lookup(self.emb_mat, self.sent2_token)
            if self.config.with_char:
                char_emb = self.build_char_embedding(self.sent2_char, self.sent2_char_len, self.char_mat,
                        is_training=is_training, reuse=reuse)
                word_emb = tf.concat([word_emb, char_emb], axis=-1)
            if self.config.apply_elmo:
                elmo_emb = tf.nn.embedding_lookup(self.elmo_mat, self.sent2_token)
                word_emb = tf.concat([word_emb, elmo_emb], axis=-1)
        return word_emb

    def build_encoder(self, index, input_lengths, *args, **kargs):

        reuse = kargs["reuse"]
        word_emb = self.build_emebdding(index, *args, **kargs)
        dropout_rate = tf.cond(self.is_training, 
                            lambda:self.config.dropout_rate,
                            lambda:0.0)

        word_emb = tf.nn.dropout(word_emb, 1-dropout_rate)
        with tf.variable_scope(self.config.scope+"_input_highway", reuse=reuse):
            input_dim = word_emb.get_shape()[-1]
            sent_repres = match_utils.multi_highway_layer(word_emb, input_dim, self.config.highway_layer_num)
            
            if self.config.rnn == "lstm":
                [sent_repres_fw, sent_repres_bw, sent_repres] = layer_utils.my_lstm_layer(sent_repres, 
                                self.config.context_lstm_dim, 
                                input_lengths=input_lengths, 
                                scope_name=self.config.scope, 
                                reuse=reuse, 
                                is_training=self.is_training,
                                dropout_rate=dropout_rate, 
                                use_cudnn=self.config.use_cudnn)

            elif self.config.rnn == "slstm":

                word_emb_proj = tf.layers.dense(word_emb, 
                                        self.config.slstm_hidden_size)

                initial_hidden_states = word_emb_proj
                initial_cell_states = tf.identity(initial_hidden_states)

                [new_hidden_states, 
                new_cell_states, 
                dummynode_hidden_states] = slstm_utils.slstm_cell(self.config, 
                                    self.config.scope, 
                                    self.config.slstm_hidden_size, 
                                    input_lengths, 
                                    initial_hidden_states, 
                                    initial_cell_states, 
                                    self.config.slstm_layer_num,
                                    dropout_rate, reuse=reuse)

                sent_repres = new_hidden_states

        return sent_repres

    def build_interactor(self, ques_repres, passage_repres, 
                        passage_len, passage_mask,
                        *args, **kargs):

        reuse = kargs["reuse"]
        dropout_rate = tf.cond(self.is_training, 
                            lambda:self.config.dropout_rate,
                            lambda:0.0)
        with tf.variable_scope(self.config.scope+"_hard_attention_module", reuse=reuse):
            # batch x 1 x dim
            if self.config.hard_attention == "l2_norm":
                ques_repres = tf.layers.dense(ques_repres, self.config.hidden_units,
                                    activation=tf.nn.relu)
                ques_repres_ = tf.expand_dims(ques_repres, axis=1)
                ques_repres_ = tf.tile(ques_repres_, 
                                [1, tf.shape(passage_repres)[1], 1])

                passage_question = tf.concat([passage_repres, ques_repres_], 
                                        axis=-1)

                passage_question = tf.layers.conv1d(passage_question,
                                    filters=self.config.num_filters,
                                    kernel_size=[1],
                                    strides=1,
                                    activation=tf.nn.relu,
                                    kernel_initializer=hard_attention_utils.initializer)

                output = hard_attention_utils.hard_attention(
                            passage_question, passage_mask,
                            self.config.num_heads, dropout_rate)
            elif self.config.hard_attention == "dot":
                print("==apply dot attention of hard attention==")
                output = hard_attention_utils.alignment_hard_attention(
                            ques_repres,
                            passage_repres, passage_mask,
                            self.config.num_heads, dropout_rate)

        return output

    def build_predictor(self, repres, *args, **kargs):
        reuse = kargs["reuse"]
        num_classes = self.config.num_classes
        dropout_rate = tf.cond(self.is_training, 
                            lambda:self.config.dropout_rate,
                            lambda:0.0)

        output = tf.layers.dense(repres, 1024, activation=tf.nn.relu)
        output = tf.nn.dropout(output, (1 - dropout_rate))
        logits = tf.layers.dense(output, num_classes)
        self.logits = tf.nn.dropout(logits, (1 - dropout_rate))
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
        elif self.config.loss == "focal_loss_multi_v1":
            self.loss, _ = point_wise_loss.focal_loss_multi_v1(self.logits, self.gold_label, 
                                        self.config, *args, **kargs)

    def build_accuracy(self, *args, **kargs):
        self.pred_label = tf.argmax(self.logits, axis=-1)
        print(self.pred_label)
        correct = tf.equal(
            tf.cast(self.pred_label, tf.int32),
            tf.cast(self.gold_label, tf.int32)
        )
        self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    def build_model(self, *args, **kargs):

        sent1_encoded = self.build_encoder("question", 
                                        self.sent1_token_len, 
                                        reuse = None)
        self.sent1_encoded = hard_attention_utils.last_relevant_output(
                            sent1_encoded, self.sent1_token_len)

        self.sent2_encoded = self.build_encoder("passage", 
                                        self.sent2_token_len, 
                                        reuse = True)

        self.aggregat_repres = self.build_interactor(
                        self.sent1_encoded, 
                        self.sent2_encoded, 
                        self.sent2_token_len, 
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

