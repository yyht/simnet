import tensorflow as tf
from model.utils.bimpm import match_utils, layer_utils
from model.utils.qanet import qanet_layers
from model.utils.embed import char_embedding_utils
from loss import point_wise_loss
from base.model_template import ModelTemplate

class BiMPM(ModelTemplate):
    def __init__(self):
        super(BiMPM, self).__init__()

    def build_char_embedding(self, char_token, char_lengths, char_embedding, *args, **kargs):

        reuse = kargs["reuse"]
        if self.config.char_embedding == "lstm":
            char_emb = char_embedding_utils.lstm_char_embedding(char_token, char_lengths, char_embedding, 
                            self.config, is_training, reuse)
        elif self.config.char_embedding == "conv":
            char_emb = char_embedding_utils.conv_char_embedding(char_token, char_lengths, char_embedding, 
                            self.config, is_training, reuse)
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
        
    def build_encoder(self, index, *args, **kargs):
        
        reuse = kargs["reuse"]
        word_emb = self.build_emebdding(index, *args, **kargs)
        dropout_rate = tf.cond(self.is_training, 
                            lambda:self.config.dropout_rate,
                            lambda:0.0)

        word_emb = tf.nn.dropout(word_emb, 1-dropout_rate)
        with tf.variable_scope(self.config.scope+"_input_highway", reuse=reuse):
            input_dim = word_emb.get_shape()[-1]
            sent_repres = match_utils.multi_highway_layer(word_emb, input_dim, self.config.highway_layer_num)

        return sent_repres

    def build_interactor(self, sent1_repres, sent2_repres, sent1_len, sent2_len,
                        sent1_mask, sent2_mask, *args, **kargs):
        
        reuse = kargs["reuse"]
        input_dim = sent1_repres.get_shape()[-1]
        with tf.variable_scope(self.config.scope+"_interaction_module", reuse=reuse):
            (match_representation, match_dim) = match_utils.bilateral_match_func(sent1_repres, sent2_repres,
                            sent1_len, 
                            sent2_len, 
                            tf.cast(sent1_mask, tf.float32), 
                            tf.cast(sent2_mask, tf.float32), 
                            input_dim, self.is_training, 
                            options=self.config)

            return match_representation, match_dim

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
            logits = tf.tanh(logits)
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
        elif self.config.loss == "focal_loss_multi_v1":
            self.loss, _ = point_wise_loss.focal_loss_multi_v1(self.logits, self.gold_label, 
                                        self.config, *args, **kargs)
             

    def build_accuracy(self, *args, **kargs):
        self.pred_label = tf.argmax(self.logits, axis=-1)
        correct = tf.equal(
            tf.cast(self.pred_label, tf.int32),
            tf.cast(self.gold_label, tf.int32)
        )
        self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    def build_model(self, *args, **kargs):

        self.sent1_encoded = self.build_encoder("question", reuse = None)

        self.sent2_encoded = self.build_encoder("passage", reuse = True)

        [self.aggregat_repres, 
            match_dim] = self.build_interactor(self.sent1_encoded, 
                                    self.sent2_encoded,
                                    self.sent1_token_len, 
                                    self.sent2_token_len,
                                    self.sent1_token_mask,
                                    self.sent2_token_mask,
                                    reuse = None)

        self.build_predictor(self.aggregat_repres,
                            reuse = None,
                            match_dim = match_dim)

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








