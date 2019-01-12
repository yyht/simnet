
from copy import copy
import numpy as np
import tensorflow as tf

from model.dsmm.base_model import BaseModel
# from model.utils.dsmm.tf_common.nn_module import word_dropout
from model.utils.dsmm.tf_common.nn_module import encode, attend


class ESIMDecAttBaseModel(BaseModel):
    """
    Implementation of base model of ESIM and DecAtt
    The difference between them lies in the encoder they use.
        - ESIM: BiLSTM
        - DecAtt: timedistributed dense projection

    Reference
    Paper:
        - ESIM: Enhanced LSTM for Natural Language Inference
        - DecAtt: A Decomposable Attention Model for Natural Language Inference
    Keras:
        https://www.kaggle.com/lamdang/dl-models
    Pytorch:
        https://github.com/lanwuwei/SPM_toolkit
    """
    def __init__(self, *args, **kargs):
        super(ESIMDecAttBaseModel, self).__init__(*args, **kargs)

    def _soft_attention_alignment(self, x1, x2):
        "Align text representation with neural soft attention"
        # x1: [b, s1, d]
        # x2: [b, s2, d]
        # att: [b, s1, s2]
        att = tf.einsum("abd,acd->abc", x1, x2)
        w_att_1 = tf.nn.softmax(att, dim=1)
        w_att_2 = tf.nn.softmax(att, dim=2)
        x2_att = tf.einsum("abd,abc->acd", x1, w_att_1)
        x1_att = tf.einsum("abd,acb->acd", x2, w_att_2)
        return x1_att, x2_att


    def _esim_semantic_feature_layer(self, emb_seq_left, emb_seq_right, seq_len_left, seq_len_right, granularity="word"):
        # for sharing embedding with other sub-graph
        # #### embed
        # emb_matrix = self._get_embedding_matrix(granularity)
        # emb_seq_left = tf.nn.embedding_lookup(emb_matrix, seq_input_left)
        # emb_seq_right = tf.nn.embedding_lookup(emb_matrix, seq_input_right)
        #
        # #### dropout
        # random_seed = np.random.randint(10000000)
        # emb_seq_left = word_dropout(emb_seq_left,
        #                             training=self.is_training,
        #                             dropout=self.config["embedding_dropout"],
        #                             seed=random_seed)
        # random_seed = np.random.randint(10000000)
        # emb_seq_right = word_dropout(emb_seq_right,
        #                              training=self.is_training,
        #                              dropout=self.config["embedding_dropout"],
        #                              seed=random_seed)

        #### encode
        input_dim = self.emb_size #self.config["embedding_dim"]
        enc_seq_left = encode(emb_seq_left, method=self.config["encode_method"],
                              input_dim=input_dim,
                              params=self.config,
                              sequence_length=seq_len_left,
                              mask_zero=self.config["embedding_mask_zero"],
                              scope_name=self.model_name + "esim_enc_seq_%s" % granularity, 
                              reuse=False,
                              training=self.is_training)
        enc_seq_right = encode(emb_seq_right, method=self.config["encode_method"],
                               input_dim=input_dim,
                               params=self.config,
                               sequence_length=seq_len_right,
                               mask_zero=self.config["embedding_mask_zero"],
                               scope_name=self.model_name + "esim_enc_seq_%s" % granularity, 
                               reuse=True,
                               training=self.is_training)

        #### align
        ali_seq_left, ali_seq_right = self._soft_attention_alignment(enc_seq_left, enc_seq_right)

        #### compose
        com_seq_left = tf.concat([
            enc_seq_left,
            ali_seq_left,
            enc_seq_left * ali_seq_left,
            enc_seq_left - ali_seq_left,
        ], axis=-1)
        com_seq_right = tf.concat([
            enc_seq_right,
            ali_seq_right,
            enc_seq_right * ali_seq_right,
            enc_seq_right - ali_seq_right,
        ], axis=-1)

        input_dim = self.config["encode_dim"] * 4
        compare_seq_left = encode(com_seq_left, method=self.config["encode_method"],
                                  input_dim=input_dim,
                                  params=self.config,
                                  sequence_length=seq_len_left,
                                  mask_zero=self.config["embedding_mask_zero"],
                                  scope_name=self.model_name + "compare_seq_%s" % granularity, 
                                  reuse=False,
                                  training=self.is_training)
        compare_seq_right = encode(com_seq_right, method=self.config["encode_method"],
                                   input_dim=input_dim,
                                   params=self.config,
                                   sequence_length=seq_len_right,
                                   mask_zero=self.config["embedding_mask_zero"],
                                   scope_name=self.model_name + "compare_seq_%s" % granularity, 
                                   reuse=True,
                                   training=self.is_training)

        #### attend
        feature_dim = self.config["encode_dim"]
        att_seq_left = attend(compare_seq_left, context=None,
                              encode_dim=self.config["encode_dim"],
                              feature_dim=feature_dim,
                              attention_dim=self.config["attention_dim"],
                              method=self.config["attend_method"],
                              scope_name=self.model_name + "agg_seq_%s" % granularity,
                              reuse=False, num_heads=self.config["attention_num_heads"])
        att_seq_right = attend(compare_seq_right, context=None,
                               encode_dim=self.config["encode_dim"],
                               feature_dim=feature_dim,
                               attention_dim=self.config["attention_dim"],
                               method=self.config["attend_method"],
                               scope_name=self.model_name + "agg_seq_%s" % granularity,
                               reuse=True, num_heads=self.config["attention_num_heads"])
        return tf.concat([att_seq_left, att_seq_right], axis=-1)


    def _get_matching_features(self):
        with tf.name_scope(self.model_name):
            tf.set_random_seed(self.config["random_seed"])

            with tf.name_scope("word_network"):
                emb_seq_word_left, enc_seq_word_left, att_seq_word_left, sem_seq_word_left = \
                    self._semantic_feature_layer(
                        self.seq_word_left,
                        self.seq_len_word_left,
                        granularity="word", reuse=False)
                emb_seq_word_right, enc_seq_word_right, att_seq_word_right, sem_seq_word_right = \
                    self._semantic_feature_layer(
                        self.seq_word_right,
                        self.seq_len_word_right,
                        granularity="word", reuse=True)
                sim_word = self._esim_semantic_feature_layer(
                    emb_seq_word_left,
                    emb_seq_word_right,
                    self.seq_len_word_left,
                    self.seq_len_word_right,
                    granularity="word")

            # with tf.name_scope("char_network"):
            #     emb_seq_char_left, enc_seq_char_left, att_seq_char_left, sem_seq_char_left = \
            #         self._semantic_feature_layer(
            #             self.seq_char_left,
            #             self.seq_len_char_left,
            #             granularity="char", reuse=False)
            #     emb_seq_char_right, enc_seq_char_right, att_seq_char_right, sem_seq_char_right = \
            #         self._semantic_feature_layer(
            #             self.seq_char_right,
            #             self.seq_len_char_right,
            #             granularity="char", reuse=True)
            #     sim_char = self._esim_semantic_feature_layer(
            #         emb_seq_char_left,
            #         emb_seq_char_right,
            #         self.seq_len_char_left,
            #         self.seq_len_char_right,
            #         granularity="char")

            with tf.name_scope("matching_features"):
                matching_features_word = sim_word
                # matching_features_char = sim_char
        return matching_features_word
        # return matching_features_word, matching_features_char


class ESIM(ESIMDecAttBaseModel):
    def __init__(self, *args, **kargs):
        super(ESIM, self).__init__(*args, **kargs)

    def build_placeholder(self, config):
        config = copy(config)
        # config.update({
        #     "encode_method": "textbicurnn",
        #     "attend_method": ["ave", "max", "min", "self-attention"],

        #     # rnn
        #     "rnn_num_units": 200,
        #     "rnn_cell_type": "lstm",
        #     "rnn_num_layers": 1,

        #     # fc block
        #     "fc_type": "densenet",
        #     "fc_hidden_units": [256, 128],
        #     "fc_dropouts": [0.3, 0.3],
        # })
        super(ESIM, self).build_placeholder(config)