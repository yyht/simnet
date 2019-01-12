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
cut_tool.init_config({})

os.environ["CUDA_VISIBLE_DEVICES"] = ""

class Eval(object):
    def __init__(self, config):
        self.config = config

        with open(self.config["model_config"], "r") as frobj:
            self.model_dict = json.load(frobj)

        # self.model_config_path = self.config["model_config_path"]
        # self.vocab_path = self.config["vocab_path"]

        # if sys.version_info < (3, ):
        #     self.embedding_info = pkl.load(open(os.path.join(self.vocab_path), "rb"))
        # else:
        #     self.embedding_info = pkl.load(open(os.path.join(self.vocab_path), "rb"), 
        #                             encoding="iso-8859-1")

        # self.token2id = self.embedding_info["token2id"]
        # self.id2token = self.embedding_info["id2token"]
        # self.embedding_mat = self.embedding_info["embedding_matrix"]
        # self.extral_symbol = self.embedding_info["extra_symbol"]

    def init_model(self, model_config):

        model_name = model_config["model_name"]
        model_str = model_config["model_str"]
        model_dir = model_config["model_dir"]
        model_config_path = model_config["model_config_path"]

        FLAGS = namespace_utils.load_namespace(os.path.join(model_config_path, model_name+".json"))
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

        vocab_path = model_config["vocab_path"]

        if sys.version_info < (3, ):
            embedding_info = pkl.load(open(os.path.join(vocab_path), "rb"))
        else:
            embedding_info = pkl.load(open(os.path.join(vocab_path), "rb"), 
                                    encoding="iso-8859-1")

        if FLAGS.apply_elmo:
            FLAGS.elmo_token_emb_mat = embedding_info["elmo"]
            FLAGS.elmo_vocab_size = embedding_info["elmo"].shape[0]
            FLAGS.elmo_emb_size = embedding_info["elmo"].shape[1]

        print(len(embedding_info["token2id"]))

        token2id = embedding_info["token2id"]
        id2token = embedding_info["id2token"]
        embedding_mat = embedding_info["embedding_matrix"]
        extral_symbol = embedding_info["extra_symbol"]

        FLAGS.token_emb_mat = embedding_mat
        FLAGS.char_emb_mat = 0
        FLAGS.vocab_size = embedding_mat.shape[0]
        FLAGS.char_vocab_size = 0
        FLAGS.emb_size = embedding_mat.shape[1]
        FLAGS.extra_symbol = extral_symbol

        model.build_placeholder(FLAGS)
        model.build_op()
        model.init_step()
        model.load_model(model_dir, model_str)

        return model, token2id

    def init(self, model_config_lst):
        self.model = {}
        self.token2id = {}
        for model_name in model_config_lst:
            if model_name in self.model_dict:
                model, token2id = self.init_model(model_config_lst[model_name])
                self.model[model_name] = model
                self.token2id[model_name] = token2id

    def prepare_data(self, question, candidate_lst):
        question = data_cleaner_api.clean(question)
        question_lst = [cut_tool.cut(question)]*len(candidate_lst)
        candidate_lst = [cut_tool.cut(data_cleaner_api.clean(candidate)) for candidate in candidate_lst]

        return [question_lst, candidate_lst]

    def model_eval(self, model_name, question_lst, candidate_lst):
        
        eval_batch = get_batch_data.get_eval_batches(question_lst, 
                                    candidate_lst, 
                                    100, 
                                    self.token2id[model_name], 
                                    is_training=False)

        eval_probs = []
        for batch in eval_batch:
            logits, preds = self.model[model_name].infer(batch, mode="infer", is_training=False)
            eval_probs.extend(preds.tolist())
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

    from flask import Flask, render_template,request,json
    from flask import jsonify
    import json
    import flask
    from collections import OrderedDict
    import requests
    from pprint import pprint

    app = Flask(__name__)
    timeout = 500

    config = {}
    config["model_config"] = "/notebooks/source/simnet/model_config.json"
    # config["model_config_path"] = "/data/xuht/ai_challenge_cqmrc/model/simnet/esim/logs"
    # config["vocab_path"] = "/data/xuht/eventy_detection/emb_mat.pkl"

    model_config_lst = {}
    # model_config_lst["multi_cnn"] = {
    #     "model_name":"multi_cnn",
    #     "model_str":"multi_cnn_1535180477_0.8768854526570067_0.8080681779167869",
    #     "model_dir":"/data/xuht/test/simnet/multi_cnn/models"
    # }
    # model_config_lst["hrchy_cnn"] = {
    #     "model_name":"hrchy_cnn",
    #     "model_str":"hrchy_cnn_1535123429_0.5808155664496801_0.8067045442082665",
    #     "model_dir":"/data/xuht/test/simnet/hrchy_cnn/models"
    # }
    # model_config_lst["dsmm_esim"] = {
    #     "model_name":"dsmm_esim",
    #     "model_str":"dsmm_esim_bigru_1542595082_0.07273565738167818_0.8006622993922514.ckpt",
    #     "model_dir":"/data/xuht/nli/model/dsmm_esim_bigru/models",
    #     "model_config_path":"/data/xuht/nli/model/dsmm_esim_bigru/logs",
    #     "vocab_path":"/data/xuht/nli/model/dsmm_esim_bigru/emb_mat.pkl"
    # }
    
    model_config_lst["bimpm"] = {
        "model_name":"bimpm",
        "model_str":"bimpm_1539668100_0.26758391517067415_0.6159838411514896",
        "model_dir":"/data/xuht/ai_challenge_cqmrc/model/simnet/bimpm/models",
        "model_config_path":"/data/xuht/ai_challenge_cqmrc/model/simnet/bimpm/logs",
        "vocab_path":"/data/xuht/ai_challenge_cqmrc/nli/elmo/emb_mat.pkl",
        "elmo_token_emb_mat":"/data/xuht/duplicate_sentence/LCQMC/elmo/elmo.pkl"
    }
    # model_config_lst["lstm"] = {
    #     "model_name":"lstm",
    #     "model_str":"lstm_1535214512_0.6789901035398626_0.8285227255387739",
    #     "model_dir":"/data/xuht/test/simnet/lstm/models"
    # }
    # model_config_lst["multi_head_git"] = {
    #     "model_name":"multi_head_git",
    #     "model_str":"multi_head_git_1535124385_0.07646397827193141_0.8140909055417235",
    #     "model_dir":"/data/xuht/test/simnet/multi_head_git/models"
    # }
    
    eval_api = Eval(config)
    eval_api.init(model_config_lst)

    def infer(data):
        question = data.get("question", u"为什么头发掉得很厉害")
        candidate_lst = data.get("candidate", ['我头发为什么掉得厉害','你的头发为啥掉这么厉害', 
                    'vpn无法开通', '我的vpn无法开通', '卤面的做法,西红柿茄子素卤面怎么做好吃',
                     '茄子面条卤怎么做'])
        preds = eval_api.infer(question, candidate_lst)
        for key in preds:
            for index, item in enumerate(preds[key]):
                preds[key][index] = preds[key][index]
        return preds

    @app.route('/simnet', methods=['POST'])
    def simnet():
        data = request.get_json(force=True)
        print("=====data=====", data)
        return jsonify(infer(data))

    app.run(debug=False, host="0.0.0.0", port=7400)