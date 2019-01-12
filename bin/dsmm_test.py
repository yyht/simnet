import pickle as pkl
import tensorflow as tf
import time, json
import datetime
import numpy as np
import argparse

from random import random

import sys,os

sys.path.append("..")

# from model.bimpm.bimpm import BiMPM
# from model.esim.esim import ESIM
# from model.biblosa.biblosa import BiBLOSA
# from model.transformer.base_transformer import BaseTransformer
# from model.transformer.universal_transformer import UniversalTransformer
# from model.drcn.drcn import DRCN

from model.dsmm.bcnn import BCNN, ABCNN1, ABCNN2, ABCNN3
from model.dsmm.match_pyramid import MatchPyramid, GMatchPyramid
from model.dsmm.dssm import DSSM, CDSSM, RDSSM
from model.dsmm.decatt import DecAtt
from model.dsmm.esim import ESIM
from model.dsmm.dsmm import DSMM

from data import data_clean
from data import data_utils 
from data import get_batch_data
from data import namespace_utils

from utils import logger_utils
from collections import OrderedDict

data_clearner_api = data_clean.DataCleaner({})
cut_tool = data_utils.cut_tool_api()

def input_dict_formulation(anchor, check, label):
    Q = {}
    Q['words'] = [anchor, check]
    anchor_len = np.sum(anchor > 0, axis=-1)
    check_len = np.sum(check > 0, axis=-1)
    Q["seq_len_words"] = [anchor_len, check_len]
    Q["labels"] = label
    return Q

def prepare_data(data_path, w2v_path, vocab_path, make_vocab=True,
                elmo_w2v_path=None,
                elmo_pca=False):

    [anchor, 
    check, 
    label, 
    anchor_len, 
    check_len] = data_utils.read_data(data_path, 
                    "train", 
                    cut_tool, 
                    data_clearner_api,
                    "tab")

    if make_vocab:
        dic = data_utils.make_dic(anchor+check)
        if not elmo_w2v_path:
            data_utils.read_pretrained_embedding(w2v_path, dic, vocab_path, min_freq=3)
        else:
            data_utils.read_pretrained_elmo_embedding(w2v_path, dic, 
                            vocab_path, min_freq=3,
                            elmo_embedding_path=elmo_w2v_path,
                            elmo_pca=elmo_pca)

    if sys.version_info < (3, ):
        embedding_info = pkl.load(open(os.path.join(vocab_path), "rb"))
    else:
        embedding_info = pkl.load(open(os.path.join(vocab_path), "rb"), 
                                encoding="iso-8859-1")

    return [anchor, check, label, anchor_len, check_len, embedding_info]

def test(config):
    model_config_path = config["model_config_path"]
    FLAGS = namespace_utils.load_namespace(model_config_path)

    os.environ["CUDA_VISIBLE_DEVICES"] = config.get("gpu_id", "")
    w2v_path = config["w2v_path"]
    vocab_path = config["vocab_path"]
    test_path = config["test_path"]
    elmo_w2v_path = config.get("elmo_w2v_path", None)

    model_dir = config["model_dir"]
    model_str = config["model_str"]
    model_name = config["model"]

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    if not os.path.exists(os.path.join(model_dir, model_name)):
        os.mkdir(os.path.join(model_dir, model_name))

    if not os.path.exists(os.path.join(model_dir, model_name, "logs")):
        os.mkdir(os.path.join(model_dir, model_name, "logs"))

    if not os.path.exists(os.path.join(model_dir, model_name, "models")):
        os.mkdir(os.path.join(model_dir, model_name, "models"))

    [test_anchor, 
    test_check, 
    test_label, 
    test_anchor_len, 
    test_check_len, 
    embedding_info] = prepare_data(test_path, 
                        w2v_path, vocab_path,
                        make_vocab=False,
                        elmo_w2v_path=elmo_w2v_path,
                        elmo_pca=FLAGS.elmo_pca)

    token2id = embedding_info["token2id"]
    id2token = embedding_info["id2token"]
    embedding_mat = embedding_info["embedding_matrix"]
    extral_symbol = embedding_info["extra_symbol"]

    if FLAGS.apply_elmo:
        FLAGS.elmo_token_emb_mat = embedding_info["elmo"]
        FLAGS.elmo_vocab_size = embedding_info["elmo"].shape[0]
        FLAGS.elmo_emb_size = embedding_info["elmo"].shape[1]

    FLAGS.token_emb_mat = embedding_mat
    FLAGS.char_emb_mat = 0
    FLAGS.vocab_size = embedding_mat.shape[0]
    FLAGS.char_vocab_size = 0
    FLAGS.emb_size = embedding_mat.shape[1]
    FLAGS.extra_symbol = extral_symbol

    if FLAGS.apply_elmo:
        FLAGS.elmo_token_emb_mat = embedding_info["elmo"]
        FLAGS.elmo_vocab_size = embedding_info["elmo"].shape[0]
        FLAGS.elmo_emb_size = embedding_info["elmo"].shape[1]

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
    else:
        total_max_len = None

    model.build_placeholder(FLAGS)
    model.build_op()
    model.init_step()
    model.load_model(os.path.join(model_dir, model_name, "models"), 
                    model_str)

    test_data = get_batch_data.get_batches(test_anchor, 
            test_check, 
            test_label, FLAGS.batch_size, 
            token2id, is_training=False,
            total_max_len=total_max_len)

    test_loss, test_accuracy = 0, 0
    cnt = 0
    for index, corpus in enumerate(test_data):
        anchor, check, label = corpus
        Q = input_dict_formulation(anchor, check, label)
        try:
            [loss, logits, 
                pred_probs, accuracy] = model.infer(
                                Q, 
                                mode="test",
                                is_training=False,
                                symmetric=False)
            # print(loss)

            test_loss += loss*anchor.shape[0]
            test_accuracy += accuracy*anchor.shape[0]
            cnt += anchor.shape[0]

        except:
            continue
       
    test_loss /= float(cnt)
    test_accuracy /= float(cnt)

    print(test_loss, test_accuracy, cnt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='model name')
    parser.add_argument('--model_config', type=str, help='model config path')
    parser.add_argument('--model_dir', type=str, help='model path')
    parser.add_argument('--config_prefix', type=str, help='config path')
    parser.add_argument('--gpu_id', type=str, help='gpu id')
    parser.add_argument('--test_path', type=str, help='train data path')
    parser.add_argument('--w2v_path', type=str, help='pretrained w2v path')
    parser.add_argument('--vocab_path', type=str, help='vocab_path')
    parser.add_argument('--model_str', type=str, help='vocab_path')
    parser.add_argument('--elmo_w2v_path', type=str, help='pretrained elmo w2v path')

    args, unparsed = parser.parse_known_args()
    model_config = args.model_config

    with open(model_config, "r") as frobj:
        model_config = json.load(frobj)

    config = {}
    config["model_dir"] = args.model_dir
    config["model"] = args.model
    config["model_config_path"] = os.path.join(args.config_prefix, 
                            model_config.get(args.model, "biblosa"))

    config["gpu_id"] = args.gpu_id
    config["test_path"] = args.test_path
    config["w2v_path"] = args.w2v_path
    config["vocab_path"] = args.vocab_path
    config["model_str"] = args.model_str
    config["elmo_w2v_path"] = args.elmo_w2v_path
    
    test(config)


    










