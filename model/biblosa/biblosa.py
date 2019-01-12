import tensorflow as tf
from model.utils.embed import char_embedding_utils
from loss import point_wise_loss
from base.model_template import ModelTemplate
from model.utils.biblosa import cnn, nn, context_fusion, general, rnn, self_attn

class BiBLOSA(ModelTemplate):
    def __init__(self):
        super(BiBLOSA, self).__init__()

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

        elif index == "passage":
            word_emb = tf.nn.embedding_lookup(self.emb_mat, self.sent2_token)
            if self.config.with_char:
                char_emb = self.build_char_embedding(self.sent2_char, self.sent2_char_len, self.char_mat,
                        is_training=is_training, reuse=reuse)
                word_emb = tf.concat([word_emb, char_emb], axis=-1)
        return word_emb

    def build_encoder(self, index, input_mask, *args, **kargs):
        
        word_emb = self.build_emebdding(index, *args, **kargs)
        reuse = kargs["reuse"]
        dropout_rate = tf.cond(self.is_training, 
                            lambda:self.config.dropout_rate,
                            lambda:0.0)

        word_emb = tf.nn.dropout(word_emb, 1-dropout_rate)
        with tf.variable_scope(self.scope+'_sent_encoding', reuse=reuse):
            act_func_str = 'elu' if self.config.context_fusion_method in ['block', 'disa'] else 'relu'
            sent_repres = context_fusion.sentence_encoding_models(
                word_emb, input_mask, 
                self.config.context_fusion_method, 
                act_func_str,
                self.scope+'_ct_based_sent2vec', 
                self.config.weight_decay, 
                self.is_training, 
                1 - dropout_rate,
                block_len=self.config.block_len, 
                hn=self.config.context_lstm_dim,
                config=self.config)

        return sent_repres

    def build_interactor(self, sent1_repres, sent2_repres, sent1_len, sent2_len,
                        sent1_mask, sent2_mask, *args, **kargs):

        reuse = kargs["reuse"]
        dropout_rate = tf.cond(self.is_training, 
                            lambda:self.config.dropout_rate,
                            lambda:0.0)

        act_func = tf.nn.elu if self.config.context_fusion_method in ['block', 'disa'] else tf.nn.relu
        act_func_str = 'elu' if self.config.context_fusion_method in ['block', 'disa'] else 'relu'

        out_rep = tf.concat([sent1_repres, sent2_repres, 
                        sent1_repres - sent2_repres, 
                        sent1_repres * sent2_repres], -1)

        pre_output = act_func(nn.linear([out_rep], 
                        self.config.context_lstm_dim, 
                        True, 0., 
                        scope= self.scope+'_pre_output', 
                        squeeze=False,
                        wd=self.config.weight_decay, 
                        input_keep_prob=1 - dropout_rate,
                        is_train=self.is_training))

        pre_output1 = nn.highway_net(
            pre_output, self.config.context_lstm_dim, 
            True, 0., self.scope+'_pre_output1', 
            act_func_str, False, self.config.weight_decay, 
            1 - dropout_rate, self.is_training)

        return pre_output1

    def build_predictor(self, matched_repres, *args, **kargs):

        dropout_rate = tf.cond(self.is_training, 
                            lambda:self.config.dropout_rate,
                            lambda:0.0)

        self.logits = nn.linear([matched_repres], 
                            self.config.num_classes, 
                            True, 0., scope= self.scope+'_logits', 
                            squeeze=False,
                            wd=self.config.weight_decay, 
                            input_keep_prob=1 - dropout_rate,
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

    def build_accuracy(self, *args, **kargs):
        self.pred_label = tf.argmax(self.logits, axis=-1)
        correct = tf.equal(
            tf.cast(self.pred_label, tf.int32),
            tf.cast(self.gold_label, tf.int32)
        )
        self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    def build_model(self, *args, **kargs):

        self.sent1_encoded = self.build_encoder("question", 
                                        self.sent1_token_mask, 
                                        reuse = None)

        self.sent2_encoded = self.build_encoder("passage", 
                                        self.sent2_token_mask, 
                                        reuse = True)

        self.aggregat_repres = self.build_interactor(
                                    self.sent1_encoded, 
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