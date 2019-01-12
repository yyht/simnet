import pickle as pkl
import tensorflow as tf
import time, json
import datetime
import numpy as np
import argparse

from random import random

import sys,os

sys.path.append("..")

from model.bimpm.bimpm import BiMPM
from model.esim.esim import ESIM
from model.biblosa.biblosa import BiBLOSA
from model.transformer.base_transformer import BaseTransformer
from model.transformer.universal_transformer import UniversalTransformer

from data import data_clean
from data import data_utils 
from data import get_batch_data
from data import namespace_utils

from utils import logger_utils
from collections import OrderedDict

data_cleaner_api = data_clean.DataCleaner({})
cut_tool = data_utils.cut_tool_api()

os.environ["CUDA_VISIBLE_DEVICES"] = ""

class Eval(object):
    def __init__(self, config):
        self.config = config

        with open(self.config["model_config"], "r") as frobj:
            self.model_dict = json.load(frobj)

        self.model_config_path = self.config["model_config_path"]
        self.vocab_path = self.config["vocab_path"]

        if sys.version_info < (3, ):
            self.embedding_info = pkl.load(open(os.path.join(self.vocab_path), "rb"))
        else:
            self.embedding_info = pkl.load(open(os.path.join(self.vocab_path), "rb"), 
                                    encoding="iso-8859-1")

        self.token2id = self.embedding_info["token2id"]
        self.id2token = self.embedding_info["id2token"]
        self.embedding_mat = self.embedding_info["embedding_matrix"]
        self.extral_symbol = self.embedding_info["extra_symbol"]

    def init_model(self, model_config):

        model_name = model_config["model_name"]
        model_str = model_config["model_str"]
        model_dir = model_config["model_dir"]

        FLAGS = namespace_utils.load_namespace(os.path.join(self.model_config_path, model_name+".json"))
        if FLAGS.scope == "BiMPM":
            model = BiMPM()
        elif FLAGS.scope == "ESIM":
            model = ESIM()
        elif FLAGS.scope == "BiBLOSA":
            model = BiBLOSA()
        elif FLAGS.scope == "BaseTransformer":
            model = BaseTransformer()
        elif FLAGS.scope == "UniversalTransformer":
            model = UniversalTransformer()

        FLAGS.token_emb_mat = self.embedding_mat
        FLAGS.char_emb_mat = 0
        FLAGS.vocab_size = self.embedding_mat.shape[0]
        FLAGS.char_vocab_size = 0
        FLAGS.emb_size = self.embedding_mat.shape[1]
        FLAGS.extra_symbol = self.extral_symbol

        model.build_placeholder(FLAGS)
        model.build_op()
        model.init_step()
        model.load_model(model_dir, model_str)

        return model

    def init(self, model_config_lst):
        self.model = {}
        for model_name in model_config_lst:
            if model_name in self.model_dict:
                self.model[model_name] = self.init_model(model_config_lst[model_name])

    def prepare_data(self, question, candidate_lst):
        question = data_cleaner_api.clean(question)
        question_lst = [cut_tool.cut(question)]*len(candidate_lst)
        candidate_lst = [cut_tool.cut(data_cleaner_api.clean(candidate)) for candidate in candidate_lst]

        return [question_lst, candidate_lst]

    def model_eval(self, model_name, question_lst, candidate_lst):
        
        eval_batch = get_batch_data.get_eval_batches(question_lst, 
                                    candidate_lst, 
                                    100, 
                                    self.token2id, 
                                    is_training=False)

        eval_probs = []
        for batch in eval_batch:
            logits, preds = self.model[model_name].infer(batch, mode="infer", is_training=False)
            eval_probs.extend(list(preds[:,1]))
        return eval_probs

    def infer(self, question, candidate_lst):
        print(question, candidate_lst)
        [question_lst, 
            candidate_lst] = self.prepare_data(question, candidate_lst)
        eval_probs = {}
        for model_name in self.model:
            probs = self.model_eval(model_name, question_lst, candidate_lst)
            eval_probs[model_name] = probs
        return eval_probs

if __name__ == "__main__":
    config = {}
    config["model_config"] = "/notebooks/source/simnet/model_config.json"
    config["model_config_path"] = "/notebooks/source/simnet/configs"
    config["vocab_path"] = "/data/xuht/duplicate_sentence/LCQMC/emb_mat.pkl"

    model_config_lst = {}
    # model_config_lst["biblosa"] = {
    #     "model_name":"biblosa",
    #     "model_str":"biblosa_1535094924_1.2270020778947086_0.7294318222186782",
    #     "model_dir":"/data/xuht/test/simnet/biblosa/models"
    # }

    # model_config_lst["gru"] = {
    #     "model_name":"gru",
    #     "model_str":"gru_1535094615_1.757930796212432_0.7826136330311949",
    #     "model_dir":"/data/xuht/test/simnet/gru/models"
    # }

    model_config_lst["bimpm"] = {
        "model_name":"bimpm",
        "model_str":"bimpm_1535308428_0.07933253372508178_0.8712499981576746",
        "model_dir":"/data/xuht/test/simnet/bimpm/models"
    }
    model_config_lst["esim"] = {
        "model_name":"esim",
        "model_str":"esim_1535168381_0.05196159574817019_0.8515909083864905",
        "model_dir":"/data/xuht/test/simnet/esim/models"
    }
    eval_api = Eval(config)
    eval_api.init(model_config_lst)

    preds = eval_api.infer(u"为什么头发掉得很厉害", ['我头发为什么掉得厉害','你的头发为啥掉这么厉害', 
                    'vpn无法开通', '我的vpn无法开通', '卤面的做法,西红柿茄子素卤面怎么做好吃',
                     '茄子面条卤怎么做'])

    print(preds)
    