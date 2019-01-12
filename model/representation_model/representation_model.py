import tensorflow as tf
from model.utils.bimpm import layer_utils, match_utils
from model.utils.qanet import qanet_layers
from model.utils.embed import char_embedding_utils
from loss import pair_wise_loss
from loss import point_wise_loss
from base.model_template import ModelTemplate
from model.utils.esim import esim_utils
from model.utils.slstm import slstm_utils
from model.utils.biblosa import cnn, nn, context_fusion, general, rnn, self_attn
from model.utils.representation_model import representation_model_utils
from model.utils.transformer import base_transformer_utils
from model.utils.transformer import universal_transformer_utils
from metric import pair_wise_metric

EPSILON = 1e-8

class RepresentationModel(ModelTemplate):
    def __init__(self):
        super(RepresentationModel, self).__init__()

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
        
    def build_encoder(self, index, input_lengths, input_mask, *args, **kargs):

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
            elif self.config.rnn == "base_transformer":
                sent_repres = base_transformer_utils.transformer_encoder(
                                        sent_repres, 
                                        target_space=None, 
                                        hparams=self.config, 
                                        features=None, 
                                        make_image_summary=False)
            elif self.config.rnn == "universal_transformer":
                sent_repres, act_loss = universal_transformer_utils.universal_transformer_encoder(
                                        sent_repres, 
                                        target_space=None, 
                                        hparams=self.config, 
                                        features=None, 
                                        make_image_summary=False)
            elif self.config.rnn == "highway":
                sent_repres = sent_repres

            input_mask = tf.expand_dims(tf.cast(input_mask, tf.float32), axis=-1)
            sent_repres_sum = tf.reduce_sum(sent_repres*input_mask, axis=1)
            sent_repres_avr = tf.div(sent_repres_sum, tf.expand_dims(tf.cast(input_lengths, tf.float32)+EPSILON, -1))
            
            if self.config.metric == "Hyperbolic":
                sent_repres = tf.clip_by_norm(sent_repres_sum, 1.0-EPSILON, axes=1)
            else:
                sent_repres = sent_repres_avr

        if self.config.rnn == "universal_transformer":
            return sent_repres, act_loss
        else:
            return sent_repres

    def build_interactor(self, sent1_repres, sent2_repres, sent1_len, sent2_len,
                        sent1_mask, sent2_mask, *args, **kargs):

        if self.config.metric == "Hyperbolic":
            distance = representation_model_utils.hyperbolic_ball(
                                    sent1_repres,
                                    sent2_repres)
            distance = representation_model_utils.distance_transformation(
                                    distance, self.config)
        elif self.config.metric == "Euclidean":
            distance = pair_wise_metric.euclidean(sent1_repres, sent2_repres)
        elif self.config.metric == "Arccosine":
            distance = pair_wise_metric.arccosine(sent1_repres, sent2_repres)
        elif self.config.metric == "Cosine":
            distance = 1.0 - (pair_wise_metric.cosine(sent1_repres, sent2_repres) + 1)/2
        match_dim = 1

        return distance, match_dim

    def build_predictor(self, distance, *args, **kargs):
        match_dim = kargs["match_dim"]
        reuse = kargs["reuse"]

        with tf.variable_scope(self.config.scope+"_prediction_module", reuse=reuse):
            if self.config.metric == "Hyperbolic":
                # self.pred_probs = representation_model_utils.pair_exp_probs(distance)
                # self.logits = tf.log(self.pred_probs+EPSILON)
                self.dist = tf.expand_dims(distance, axis=1)
                self.pred_probs = tf.concat([self.dist,
                                    self.config.loss_margin - self.dist], axis=1)
                self.logits = tf.log(self.pred_probs+EPSILON)
                print("====size of pred_probs====", self.pred_probs.get_shape())
            elif self.config.metric in ["Euclidean", "Arccosine", "Cosine"]:
                self.dist = tf.expand_dims(distance, axis=1)
                self.pred_probs = tf.concat([self.dist, 1 - self.dist], axis=-1)
                self.logits = tf.log(self.pred_probs+EPSILON)
                print("====size of pred_probs====", self.pred_probs.get_shape())

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
        elif self.config.loss == "contrastive_loss":
            if self.config.metric in ["Hyperbolic"]:
                self.loss = pair_wise_loss.hyper_contrastive_loss(self.dist, 
                                                        self.gold_label,
                                                        self.config,
                                                        is_quardic=True)
            elif self.config.metric in ["Euclidean", "Arccosine", "Cosine"]:
                self.loss = pair_wise_loss.contrastive_loss(self.dist, 
                                                        self.gold_label,
                                                        self.config,
                                                        is_quardic=True)
                
        if self.config.get("weight_decay", None):
            model_vars = tf.trainable_variables() 
            lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in model_vars \
                                if 'bias' not in v.name ])
            lossL2 *= self.config.weight_decay
            self.loss += lossL2
        if self.config.rnn == "universal_transformer":
            print("====encoder type====", self.config.rnn)
            self.loss += (self.act_loss/2.0)

        # if self.config.metric == "Hyperbolic":
        #     for var in trainable_vars:
        #         var_norm = representation_model_utils.var_nrom(var)
        #         self.loss += self.config.lagrangian * (var_norm-1)

    def build_accuracy(self, *args, **kargs):
        self.pred_label = tf.argmax(self.logits, axis=-1)
        correct = tf.equal(
            tf.cast(self.pred_label, tf.int32),
            tf.cast(self.gold_label, tf.int32)
        )
        self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    def build_model(self, *args, **kargs):

        sent1_repres = self.build_encoder("question", 
                                        self.sent1_token_len, 
                                        self.sent1_token_mask,
                                        reuse = None)
        sent2_repres = self.build_encoder("passage", 
                                        self.sent2_token_len,
                                        self.sent2_token_mask,
                                        reuse = True)

        if self.config.rnn == "universal_transformer":
            self.sent1_repres = sent1_repres[0]
            self.sent2_repres = sent2_repres[0]
            self.act_loss = sent1_repres[1] + sent2_repres[1]
        else:
            self.sent1_repres = sent1_repres
            self.sent2_repres = sent2_repres
        
        [self.aggregat_repres, 
            match_dim] = self.build_interactor(self.sent1_repres, 
                                    self.sent2_repres,
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