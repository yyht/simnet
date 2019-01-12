import tensorflow as tf
from model.utils.bimpm import layer_utils, match_utils
from model.utils.qanet import qanet_layers
from model.utils.embed import char_embedding_utils
from loss import point_wise_loss
from base.model_template import ModelTemplate
from model.utils.esim import esim_utils
from model.utils.slstm import slstm_utils
from model.utils.biblosa import cnn, nn, context_fusion, general, rnn, self_attn
from model.utils.drcn import drcn_utils
import numpy as np
EPSILON = 1e-8

class DRCN(ModelTemplate):
    def __init__(self):
        super(DRCN, self).__init__()

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

    def build_encoder(self, sent_repres, input_lengths, *args, **kargs):

        reuse = kargs["reuse"]
        dropout_rate = tf.cond(self.is_training, 
                            lambda:self.config.dropout_rate,
                            lambda:0.0)

        with tf.variable_scope(self.config.scope+"_input_highway", reuse=reuse):
            
            if self.config.rnn == "lstm":
                [sent_repres_fw, sent_repres_bw, sent_repres] = layer_utils.my_lstm_layer(sent_repres, 
                                self.config.context_lstm_dim, 
                                input_lengths=input_lengths, 
                                scope_name=self.config.scope, 
                                reuse=reuse, 
                                is_training=self.is_training,
                                dropout_rate=dropout_rate, 
                                use_cudnn=self.config.use_cudnn,
                                lstm_type=self.config.lstm_type)
                match_dim = self.config.context_lstm_dim * 2

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
                match_dim = self.config.slstm_hidden_size * 2
                sent_repres = new_hidden_states

        return sent_repres, match_dim

    def auto_encoder(self, sent_repres, *args, **kargs):
        reuse = kargs["reuse"]
        sent_repres = sent_repres
        return sent_repres

    def build_interactor(self, sent1_emb, sent2_emb, 
                        sent1_len, sent2_len,
                        sent1_mask, sent2_mask, 
                        *args, **kargs):

        num_lstm_layers = kargs["num_lstm_layers"]
        dropout_rate = tf.cond(self.is_training, 
                            lambda:self.config.dropout_rate,
                            lambda:0.0)

        input_dim = sent1_emb.get_shape()[-1]
        with tf.variable_scope(self.config.scope+"_embed_hishway"):
            sent1_repres = match_utils.multi_highway_layer(sent1_emb, 
                                        input_dim, 
                                        self.config.highway_layer_num)
            tf.get_variable_scope().reuse_variables()
            sent2_repres = match_utils.multi_highway_layer(sent2_emb, 
                                        input_dim, 
                                        self.config.highway_layer_num)
        match_dim = self.emb_size
        for i in range(num_lstm_layers):
            with tf.variable_scope(self.config.scope+"_densely_co_attentive_{}".format(i), 
                                    reuse=None):
                sent1_repres_, match_dim_ = self.build_encoder(sent1_repres, sent1_len, 
                                                reuse=None)
                sent2_repres_, match_dim_ = self.build_encoder(sent2_repres, sent1_len, 
                                                reuse=True)
                match_dim += match_dim_
                print("===before=====", i, sent1_repres_.get_shape(), sent2_repres_.get_shape())
                if self.config.get("co_attention", None):
                    [query_attention, 
                    context_attention] = drcn_utils.query_context_attention(
                                            sent1_repres_, sent2_repres_, 
                                            sent1_len, sent2_len, 
                                            sent1_mask, sent2_mask, dropout_rate,
                                            self.config.scope, reuse=None)
                    
                    sent1_repres = tf.concat([sent1_repres_, query_attention, sent1_repres],
                                            axis=-1)
                    sent2_repres = tf.concat([sent2_repres_, context_attention, sent2_repres],
                                            axis=-1)
                    match_dim += match_dim_
                else:
                    sent1_repres = tf.concat([sent1_repres_, sent1_repres],
                                            axis=-1)
                    sent2_repres = tf.concat([sent2_repres_, sent2_repres],
                                            axis=-1)

                print("====i====", sent1_repres.get_shape(), sent2_repres.get_shape())
                if np.mod(i+1, 2) == 0 and self.config.with_auto_encoding:
                    sent1_repres = self.auto_encoder(sent1_repres, reuse=None)
                    sent2_repres = self.auto_encoder(sent2_repres, reuse=True)

                if self.config.recurrent_layer_norm:
                    sent1_repres = tf.contrib.layers.layer_norm(sent1_repres, reuse=None, scope="lstm_layer_norm")
                    sent2_repres = tf.contrib.layers.layer_norm(sent2_repres, reuse=True, scope="lstm_layer_norm")
        
        mask_q = tf.expand_dims(sent1_mask, -1)
        mask_c = tf.expand_dims(sent2_mask, -1)

        v_1_max = tf.reduce_max(qanet_layers.mask_logits(sent1_repres, mask_q), axis=1)
        v_2_max = tf.reduce_max(qanet_layers.mask_logits(sent2_repres, mask_c), axis=1)

        v = tf.concat([v_1_max, v_2_max,
                        v_1_max*v_2_max,
                        v_1_max-v_2_max, 
                        tf.abs(v_1_max-v_2_max)],
                        axis=-1)
        v = tf.nn.dropout(v, 1 - dropout_rate)
        match_dim = match_dim * 5

        return v_1_max, v_2_max, v, match_dim

    def build_predictor(self, matched_repres, *args, **kargs):
        match_dim = kargs["match_dim"]
        reuse = kargs["reuse"]
        num_classes = self.config.num_classes
        dropout_rate = tf.cond(self.is_training, 
                            lambda:self.config.dropout_rate,
                            lambda:0.0)

        with tf.variable_scope(self.config.scope+"_prediction_module", reuse=reuse):
            #========Prediction Layer=========
            # match_dim = 4 * self.options.aggregation_lstm_dim
            w_0 = tf.get_variable("w_0", [match_dim, match_dim/2], dtype=tf.float32)
            b_0 = tf.get_variable("b_0", [match_dim/2], dtype=tf.float32)
            w_1 = tf.get_variable("w_1", [match_dim/2, num_classes],dtype=tf.float32)
            b_1 = tf.get_variable("b_1", [num_classes],dtype=tf.float32)

            # if is_training: match_representation = tf.nn.dropout(match_representation, (1 - options.dropout_rate))
            logits = tf.matmul(matched_repres, w_0) + b_0
            logits = tf.nn.relu(logits)
            logits = tf.nn.dropout(logits, (1 - dropout_rate))
            
            self.logits = tf.matmul(logits, w_1) + b_1
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

        if self.config.get("weight_decay", None):
            for var in set(tf.get_collection('reg_vars', self.scope)):
                weight_decay = tf.multiply(tf.nn.l2_loss(var), self.config.weight_decay,
                                          name="{}-wd".format('-'.join(str(var.op.name).split('/'))))
                self.loss += weight_decay

    def build_accuracy(self, *args, **kargs):
        self.pred_label = tf.argmax(self.logits, axis=-1)
        print(self.pred_label)
        correct = tf.equal(
            tf.cast(self.pred_label, tf.int32),
            tf.cast(self.gold_label, tf.int32)
        )
        self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    def build_model(self, *args, **kargs):

        self.sent1_embed = self.build_emebdding("question", reuse=None)
        self.sent2_embed = self.build_emebdding("passage", reuse=True)
        
        [self.sent1_repres, 
        self.sent2_repres, 
        self.aggregat_repres,
        match_dim] = self.build_interactor(self.sent1_embed, 
                                    self.sent2_embed,
                                    self.sent1_token_len, 
                                    self.sent2_token_len,
                                    self.sent1_token_mask,
                                    self.sent2_token_mask,
                                    num_lstm_layers = self.config.num_lstm_layers)

        self.build_predictor(self.aggregat_repres,
                            reuse = None,
                            match_dim = match_dim)

        # drcn_utils.add_reg_without_bias(self.config.scope)

    def get_feed_dict(self, sample_batch, *args, **kargs):
        [sent1_token, sent2_token, gold_label] = sample_batch

        feed_dict = {
            self.sent1_token: sent1_token,
            self.sent2_token: sent2_token,
            self.gold_label:gold_label,
            self.learning_rate: kargs["learning_rate"],
            self.is_training:kargs["is_training"]
        }
        return feed_dict