import tensorflow as tf
from model.utils.bimpm import layer_utils, match_utils
from model.utils.qanet import qanet_layers
from model.utils.embed import char_embedding_utils
from loss import point_wise_loss
from base.model_template import ModelTemplate
from model.utils.decathlon import decathlon_utils
from model.utils.slstm import slstm_utils
from model.utils.biblosa import cnn, nn, context_fusion, general, rnn, self_attn

EPSILON = 1e-8

class Decathlon(ModelTemplate):
    def __init__(self):
        super(Decathlon, self).__init__()

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
        elif index == "answer":
            word_emb = tf.nn.embedding_lookup(self.emb_mat, self.ans_token)
            if self.config.with_char:
                char_emb = self.build_char_embedding(self.ans_char, self.ans_char_len, self.char_mat,
                        is_training=is_training, reuse=reuse)
                word_emb = tf.concat([word_emb, char_emb], axis=-1)

        return word_emb
        
    def build_encoder(self, index, input_lengths, input_mask, 
                    *args, **kargs):

        reuse = kargs["reuse"]
        word_emb = self.build_emebdding(index, *args, **kargs)
        dropout_rate = tf.cond(self.is_training, 
                            lambda:self.config.dropout_rate,
                            lambda:0.0)

        word_emb = tf.nn.dropout(word_emb, 1-dropout_rate)
        with tf.variable_scope(self.config.scope+"_input_highway", reuse=reuse):
            input_dim = word_emb.get_shape()[-1]
            sent_repres = match_utils.multi_highway_layer(word_emb, input_dim, self.config.highway_layer_num)
            
            [_, _, sent_repres] = layer_utils.my_lstm_layer(sent_repres, 
                            self.config.context_lstm_dim, 
                            input_lengths=input_lengths, 
                            scope_name=self.config.scope, 
                            reuse=reuse, 
                            is_training=self.is_training,
                            dropout_rate=dropout_rate, 
                            use_cudnn=self.config.use_cudnn)

            sent_repres = tf.layers.dense(sent_repres, 
                            self.config.context_lstm_dim*2, 
                            activation=tf.nn.relu) + sent_repres

            ignore_padding = (1 - input_mask)
            ignore_padding = decathlon_utils.attention_bias_ignore_padding(ignore_padding)
            encoder_self_attention_bias = ignore_padding

            output = decathlon_utils.multihead_attention_texar(sent_repres, 
                            memory=None, 
                            memory_attention_bias=encoder_self_attention_bias,
                            num_heads=self.config.num_heads, 
                            num_units=None, 
                            dropout_rate=dropout_rate, 
                            scope="multihead_attention")

            output = tf.layers.dense(output, 
                            self.config.context_lstm_dim*2, 
                            activation=tf.nn.relu) + output

            output = qanet_layers.layer_norm(output, 
                                scope = "layer_norm", 
                                reuse = reuse)

        return sent_repres

    def build_multi_attention(self, sent1_repres, sent2_repres, sent1_len, sent2_len,
                        sent1_mask, sent2_mask, *args, **kargs):
        reuse = kargs["reuse"]
        dropout_rate = tf.cond(self.is_training, 
                            lambda:self.config.dropout_rate,
                            lambda:0.0)

        with tf.variable_scope(self.config.scope+"_multi_attention", reuse=reuse):

            c2q, q2c = decathlon_utils.interaction_attention(sent1_repres, sent2_repres, 
                        sent1_mask, sent2_mask, 
                        self.config.attn_lst, 
                        dropout_rate, 
                        self.config.scope)

            return c2q, q2c

    def build_attention_aggregation(self, coattention, context, context_len, context_mask, 
                        sent1_repres, sent2_repres, 
                        sent1_len, sent2_len,
                        sent1_mask, sent2_mask, 
                        *args, **kargs):
        reuse = kargs["reuse"]
        dropout_rate = tf.cond(self.is_training, 
                            lambda:self.config.dropout_rate,
                            lambda:0.0)

        context_fusion = []
        with tf.variable_scope(self.config.scope+"_aggerate_attention", reuse=reuse):

            for i in range(len(self.config.attn_lst)):
                context_f = tf.concat([context, coattention[i]], axis=-1)
 
                with tf.variable_scope(self.config.scope+"_attn_fusion_{}".format(i), 
                        reuse=reuse):

                    [_, _, context_f] = layer_utils.my_lstm_layer(context_f, 
                            self.config.context_lstm_dim, 
                            input_lengths=context_len, 
                            scope_name=self.config.scope+"_context", 
                            reuse=None, 
                            is_training=self.is_training,
                            dropout_rate=dropout_rate, 
                            use_cudnn=self.config.use_cudnn)

                    context_fusion.append(context_f)

            # batch x 4 x len x dim
            context_fusion = tf.stack(context_fusion, axis=1) 

            with tf.variable_scope(self.config.scope+"_attention_fusion", reuse=reuse): 

                context_fusion = decathlon_utils.attention_fusion(
                                    context_fusion, 
                                    context_maks, 
                                    self.config.scope+"_context_fusion",
                                    reuse=reuse)

            return context_fusion

    def build_compression(self, context_fusion, context_mask, context_len,
                        scope_name, 
                        *args, **kargs):

        reuse = kargs["reuse"]
        dropout_rate = tf.cond(self.is_training, 
                            lambda:self.config.dropout_rate,
                            lambda:0.0)

        with tf.variable_scope(self.config.scope+"_compression_"+scope_name, 
                                reuse=reuse): 

            ignore_padding = (1 - context_mask)
            ignore_padding = decathlon_utils.attention_bias_ignore_padding(ignore_padding)
            encoder_self_attention_bias = ignore_padding

            context_repres = decathlon_utils.multihead_attention_texar(context_fusion, 
                            memory=None, 
                            memory_attention_bias=encoder_self_attention_bias,
                            num_heads=self.config.num_heads, 
                            num_units=None, 
                            dropout_rate=dropout_rate, 
                            scope="context")

            context_repres = tf.layers.dense(context_repres, 
                            self.config.context_lstm_dim*2, 
                            activation=tf.nn.relu) + context_repres

            context_repres = qanet_layers.layer_norm(context_repres, 
                                scope = "layer_norm", 
                                reuse = reuse)

            [_, _, context_repres] = layer_utils.my_lstm_layer(context_repres, 
                            self.config.context_lstm_dim, 
                            input_lengths=input_lengths, 
                            scope_name=self.config.scope, 
                            reuse=reuse, 
                            is_training=self.is_training,
                            dropout_rate=dropout_rate, 
                            use_cudnn=self.config.use_cudnn)

            return context_repres

    def build_decoder(self, query_repres, query_mask,
                    context_repres, context_mask, 
                    answer_repres, answer_mask,
                    *args, **kargs):

        reuse = kargs["reuse"]
        dropout_rate = tf.cond(self.is_training, 
                            lambda:self.config.dropout_rate,
                            lambda:0.0)

        if self.config.mrc == "span":
            




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
        print(self.pred_label)
        correct = tf.equal(
            tf.cast(self.pred_label, tf.int32),
            tf.cast(self.gold_label, tf.int32)
        )
        self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    def build_model(self, *args, **kargs):

        self.sent1_encoded = self.build_encoder("question", 
                                        self.sent1_token_len, 
                                        reuse = None)

        self.sent2_encoded = self.build_encoder("passage", 
                                        self.sent2_token_len, 
                                        reuse = True)

        [self.sent1_repres, 
        self.sent2_repres, 
        self.aggregat_repres,
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