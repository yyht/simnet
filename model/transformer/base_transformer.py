import tensorflow as tf
from model.utils.embed import char_embedding_utils
from loss import point_wise_loss
from base.model_template import ModelTemplate
from model.utils.transformer import base_transformer_utils
from model.utils.esim import esim_utils
from model.utils.slstm import slstm_utils
from model.utils.biblosa import cnn, nn, context_fusion, general, rnn, self_attn

class BaseTransformer(ModelTemplate):
    def __init__(self):
        super(BaseTransformer, self).__init__()

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

    def build_encoder(self, index, input_mask, *args, **kargs):
        reuse = kargs["reuse"]
        word_emb = self.build_emebdding(index, *args, **kargs)
        dropout_rate = tf.cond(self.is_training, 
                            lambda:self.config.dropout_rate,
                            lambda:0.0)

        word_emb = tf.nn.dropout(word_emb, 1-dropout_rate)
        input_mask = tf.cast(input_mask, tf.float32)
        input_mask = tf.expand_dims(input_mask, axis=-1) # batch_size x seq_len x 1
        word_emb *= input_mask

        word_emb = tf.layers.dense(word_emb, self.config.hidden_size)
    
        with tf.variable_scope(self.config.scope+"_transformer_encoder", 
                    reuse=reuse):
            encoder_output = base_transformer_utils.transformer_encoder(word_emb, 
                                        target_space=None, 
                                        hparams=self.config, 
                                        features=None, 
                                        losses=None)

            input_mask = tf.squeeze(input_mask, axis=-1)
            encoder_output = self_attn.multi_dimensional_attention(
                encoder_output, input_mask, 'multi_dim_attn_for_%s' % self.config.scope,
                1 - dropout_rate, self.is_training, self.config.weight_decay, "relu")
            
            return encoder_output

    def build_interactor(self, sent1_repres, sent2_repres, sent1_len, sent2_len,
                        sent1_mask, sent2_mask, *args, **kargs):
        reuse = kargs["reuse"]
        dropout_rate = tf.cond(self.is_training, 
                            lambda:self.config.dropout_rate,
                            lambda:0.0)

        act_func = tf.nn.relu
        act_func_str = 'relu'

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
        reuse = kargs["reuse"]
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

        self.aggregat_repres = self.build_interactor(self.sent1_encoded, 
                                    self.sent2_encoded,
                                    self.sent1_token_len, 
                                    self.sent2_token_len,
                                    self.sent1_token_mask,
                                    self.sent2_token_mask,
                                    reuse = None)

        self.build_predictor(self.aggregat_repres,
                            reuse = None)

        print("List of Variables:")
        for v in tf.trainable_variables():
            print(v.name)

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