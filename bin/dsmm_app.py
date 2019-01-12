import pickle as pkl
import tensorflow as tf
import time, json
import datetime
import numpy as np
import argparse

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from random import random

import sys,os

sys.path.append("..")

from model.dsmm.bcnn import BCNN, ABCNN1, ABCNN2, ABCNN3
from model.dsmm.match_pyramid import MatchPyramid, GMatchPyramid
from model.dsmm.dssm import DSSM, CDSSM, RDSSM
from model.dsmm.decatt import DecAtt
from model.dsmm.esim import ESIM
from model.dsmm.dsmm import DSMM
from model.diin.diin import DIIN

from data import data_clean
from data import data_utils 
from data import get_batch_data
from data import namespace_utils

from utils import logger_utils
from collections import OrderedDict

data_cleaner_api = data_clean.DataCleaner({})
cut_tool = data_utils.cut_tool_api()

# cut_tool.init_config({
#             "user_dict":"/data/xuht/eventy_detection/inference_data/project_entity.txt"})

cut_tool.init_config({})

cut_tool.build_tool()

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def input_dict_formulation(anchor, check, label):
    Q = {}
    Q['words'] = [anchor, check]
    anchor_len = np.sum(anchor > 0, axis=-1)
    check_len = np.sum(check > 0, axis=-1)
    Q["seq_len_words"] = [anchor_len, check_len]
    Q["labels"] = label
    return Q

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

        print(model_config, "==model config==")

        try:
            model_name = model_config["output_folder_name"]
        except:
            model_name = model_config["model_name"]

        model_str = model_config["model_str"]
        model_dir = model_config["model_dir"]
        model_config_path = model_config["model_config_path"]

        FLAGS = namespace_utils.load_namespace(os.path.join(model_config_path, model_name+".json"))
        if FLAGS.scope == "BCNN":
            model = BCNN()
            total_max_len = FLAGS.max_seq_len_word
        elif FLAGS.scope == "ABCNN1":
            model = ABCNN1()
            total_max_len = FLAGS.max_seq_len_word
        elif FLAGS.scope == "ABCNN2":
            model = ABCNN2()
            total_max_len = FLAGS.max_seq_len_word
        elif FLAGS.scope == "ABCNN3":
            model = ABCNN3()
            total_max_len = FLAGS.max_seq_len_word
        elif FLAGS.scope == "MatchPyramid":
            model = MatchPyramid()
            total_max_len = FLAGS.max_seq_len_word
        elif FLAGS.scope == "GMatchPyramid":
            model = GMatchPyramid()
            total_max_len = FLAGS.max_seq_len_word
        elif FLAGS.scope == "DSSM":
            model = DSSM()
            total_max_len = FLAGS.max_seq_len_word
        elif FLAGS.scope == "CDSSM":
            model = CDSSM()
            total_max_len = FLAGS.max_seq_len_word
        elif FLAGS.scope == "RDSSM":
            model = RDSSM()
            total_max_len = FLAGS.max_seq_len_word
        elif FLAGS.scope == "DecAtt":
            model = DecAtt()
            total_max_len = FLAGS.max_seq_len_word
        elif FLAGS.scope == "DSMM_ESIM":
            model = ESIM()
            total_max_len = FLAGS.max_seq_len_word
        elif FLAGS.scope == "DSMM":
            model = DSMM()
            total_max_len = FLAGS.max_seq_len_word
        elif FLAGS.scope == "DIIN":
            model = DIIN()
            total_max_len = FLAGS.max_seq_len_word
        else:
            total_max_len = None
        self.total_max_len = total_max_len

        vocab_path = model_config["vocab_path"]

        if sys.version_info < (3, ):
            embedding_info = pkl.load(open(os.path.join(vocab_path), "rb"))
        else:
            embedding_info = pkl.load(open(os.path.join(vocab_path), "rb"), 
                                    encoding="iso-8859-1")

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
        print(question_lst[0], "===question===", candidate_lst[0], "===candidate===")
        return [question_lst, candidate_lst]

    def model_eval(self, model_name, question_lst, candidate_lst):
        
        eval_batch = get_batch_data.get_eval_batches(question_lst, 
                                    candidate_lst, 
                                    100, 
                                    self.token2id[model_name], 
                                    is_training=False,
                                    total_max_len=self.total_max_len)

        eval_probs = []
        for batch in eval_batch:
            anchor, check, label = batch
            Q = input_dict_formulation(anchor, check, label)
            logits, preds = self.model[model_name].infer(Q, mode="infer", 
                                is_training=False, symmetric=False)
            eval_probs.extend(preds.tolist())
        return eval_probs

    def infer(self, question, candidate_lst):
        
        [question_lst, 
            candidate_lst] = self.prepare_data(question, candidate_lst)
        print(question, candidate_lst)
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
    # model_config_lst["bimpm"] = {
    #     "model_name":"bimpm",
    #     "model_str":"bimpm_1535308428_0.07933253372508178_0.8712499981576746",
    #     "model_dir":"/data/xuht/test/simnet/bimpm/models"
    # }
    
    # model_config_lst["dsmm_esim"] = {
    #     "model_name":"dsmm_esim",
    #     "model_str":"dsmm_esim_1539569329_0.3109117832124446_0.6126619553751117",
    #     "model_dir":"/data/xuht/ai_challenge_cqmrc/model/simnet/dsmm_esim/models",
    #     "model_config_path":"/data/xuht/ai_challenge_cqmrc/model/simnet/dsmm_esim/logs",
    #     "vocab_path":"/data/xuht/ai_challenge_cqmrc/nli/dsmm/emb_mat.pkl"
    # }

    # model_config_lst["dsmm_esim"] = {
    #     "model_name":"dsmm_esim",
    #     "model_str":"dsmm_esim_1539657279_0.0410678406940615_0.7880183620501359",
    #     "model_dir":"/data/xuht/eventy_detection/inference_data/nli/model/dsmm_esim/models",
    #     "model_config_path":"/data/xuht/eventy_detection/inference_data/nli/model/dsmm_esim/logs",
    #     "vocab_path":"/data/xuht/eventy_detection/inference_data/nli/nli_dsmm/emb_mat.pkl"
    # }

    model_config_lst["dsmm_esim"] = {
        "model_name":"dsmm_esim",
        "model_str":"dsmm_esim_bigru_1542595082_0.07273565738167818_0.8006622993922514",
        "model_dir":"/data/xuht/nli/model/dsmm_esim_bigru/models",
        "model_config_path":"/data/xuht/nli/model/dsmm_esim_bigru/logs",
        "vocab_path":"/data/xuht/nli/model/dsmm_esim_bigru/emb_mat.pkl"
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