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

data_clearner_api = data_clean.DataCleaner({})
# cut_tool = data_utils.cut_tool_api()
# cut_tool.build_tool()

def prepare_data(data_path, w2v_path, vocab_path, make_vocab=True,
                elmo_w2v_path=None,
                elmo_pca=False,
                data_type="event_nli"):

    [anchor, 
    check, 
    label, 
    anchor_len, 
    check_len] = data_utils.read_data(data_path, 
                    "train", 
                    cut_tool, 
                    data_clearner_api,
                    "tab",
                    data_type=data_type)

    if make_vocab:
        dic = data_utils.make_dic(anchor+check)
        if not elmo_w2v_path:
            data_utils.read_pretrained_embedding(w2v_path, dic, vocab_path, min_freq=5)
        else:
            data_utils.read_pretrained_elmo_embedding(w2v_path, dic, 
                            vocab_path, min_freq=5,
                            elmo_embedding_path=elmo_w2v_path,
                            elmo_pca=elmo_pca)

    if sys.version_info < (3, ):
        embedding_info = pkl.load(open(os.path.join(vocab_path), "rb"))
    else:
        embedding_info = pkl.load(open(os.path.join(vocab_path), "rb"), 
                                encoding="iso-8859-1")

    return [anchor, check, label, anchor_len, check_len, embedding_info]

def evaluate(pred_label, true_label, target_label_id):
    count = 0
    for pred, true in zip(pred_label, true_label):
        if pred == true and true == target_label_id:
            count += 1
    precision = count / float(np.sum(np.asarray(pred_label)==target_label_id)+1e-10)
    recall = count / float(np.sum(np.asarray(true_label)==target_label_id)+1e-10)
    f1 = 2 * precision * recall / (recall + precision+1e-10)
    return precision, recall, f1

def input_dict_formulation(anchor, check, label):
    Q = {}
    Q['words'] = [anchor, check]
    anchor_len = np.sum(anchor > 0, axis=-1)
    check_len = np.sum(check > 0, axis=-1)
    Q["seq_len_words"] = [anchor_len, check_len]
    Q["labels"] = label
    return Q

def train(config):
    model_config_path = config["model_config_path"]
    FLAGS = namespace_utils.load_namespace(model_config_path)

    os.environ["CUDA_VISIBLE_DEVICES"] = config.get("gpu_id", "")
    train_path = config["train_path"]
    w2v_path = config["w2v_path"]
    vocab_path = config["vocab_path"]
    dev_path = config["dev_path"]
    elmo_w2v_path = config.get("elmo_w2v_path", None)

    model_dir = config["model_dir"]
    model_name = config["model"]

    model_dir = config["model_dir"]
    try:
        model_name = FLAGS["output_folder_name"]
    except:
        model_name = config["model"]
    print(model_name, "====model name====")

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    if not os.path.exists(os.path.join(model_dir, model_name)):
        os.mkdir(os.path.join(model_dir, model_name))

    if not os.path.exists(os.path.join(model_dir, model_name, "logs")):
        os.mkdir(os.path.join(model_dir, model_name, "logs"))

    if not os.path.exists(os.path.join(model_dir, model_name, "models")):
        os.mkdir(os.path.join(model_dir, model_name, "models"))

    logger = logger_utils.get_logger(os.path.join(model_dir, model_name, "logs","log.info"))
    FLAGS.vocab_path = vocab_path
    json.dump(FLAGS, open(os.path.join(model_dir, model_name, "logs", model_name+".json"), "w"))

    [train_anchor, 
    train_check, 
    train_label, 
    train_anchor_len, 
    train_check_len, 
    embedding_info] = prepare_data(train_path, 
                        w2v_path, vocab_path,
                        make_vocab=True,
                        elmo_w2v_path=elmo_w2v_path,
                        elmo_pca=FLAGS.elmo_pca,
                        data_type=config["data_type"])

    [dev_anchor, 
    dev_check, 
    dev_label, 
    dev_anchor_len, 
    dev_check_len, 
    embedding_info] = prepare_data(dev_path, 
                        w2v_path, vocab_path,
                        make_vocab=False,
                        elmo_w2v_path=elmo_w2v_path,
                        elmo_pca=FLAGS.elmo_pca,
                        data_type=config["data_type"])

    token2id = embedding_info["token2id"]
    id2token = embedding_info["id2token"]
    embedding_mat = embedding_info["embedding_matrix"]
    extral_symbol = embedding_info["extra_symbol"]

    logger.info("==vocab size {}".format(len(token2id)))
    logger.info("vocab path {}".format(vocab_path))

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
    elif FLAGS.scope == "DIIN":
        model = DIIN()
        total_max_len = FLAGS.max_seq_len_word
    else:
        total_max_len = None
    # elif FLAGS.scope == "UniversalTransformer":
    #     model = UniversalTransformer()
    # elif FLAGS.scope == "DRCN":
    #     model = DRCN()
    # elif FLAGS.scope == "RepresentationModel":
    #     model = RepresentationModel()

    model.build_placeholder(FLAGS)
    model.build_op()
    model.init_step()

    print("========begin to train=========")

    best_dev_accuracy, best_dev_loss, best_dev_f1 = 0, 100, 0

    cnt = 0
    toleration_cnt = 0
    toleration = 10
    for epoch in range(FLAGS.max_epochs):
        train_loss, train_accuracy = 0, 0
        train_data = get_batch_data.get_batches(train_anchor, 
            train_check, 
            train_label, FLAGS.batch_size, 
            token2id, is_training=True,
            total_max_len=total_max_len)

        cnt = 0
        train_accuracy_score, train_precision_score, train_recall_score = 0, 0 ,0
        train_label_lst, train_true_lst = [], []
        
        for index, corpus in enumerate(train_data):
            anchor, check, label = corpus
            Q = input_dict_formulation(anchor, check, label)
            # try:
            [loss, _, global_step, 
            accuracy, preds] = model.step(
                                Q, is_training=True, 
                                symmetric=False)

            train_label_lst += np.argmax(preds, axis=-1).tolist()
            train_true_lst += label.tolist()

            train_loss += loss*anchor.shape[0]
            train_accuracy += accuracy*anchor.shape[0]
            cnt += anchor.shape[0]
            # except:
            #     continue

        train_loss /= float(cnt)

        train_accuracy = accuracy_score(train_true_lst, train_label_lst)
        train_recall = recall_score(train_true_lst, train_label_lst, average='macro')
        train_precision = precision_score(train_true_lst, train_label_lst, average='macro')
        train_f1 = f1_score(train_true_lst, train_label_lst, average='macro')

        # [train_precision, 
        # train_recall, 
        # train_f1] = evaluate(train_label_lst, train_true_lst, 1)

        info = OrderedDict()
        info["epoch"] = str(epoch)
        info["train_loss"] = str(train_loss)
        info["train_accuracy"] = str(train_accuracy)
        info["train_f1"] = str(train_f1)

        logger.info("epoch\t{}\ttrain\tloss\t{}\taccuracy\t{}\tf1\t{}".format(epoch, train_loss, 
                                                                train_accuracy, train_f1))

        dev_data = get_batch_data.get_batches(dev_anchor, 
            dev_check, 
            dev_label, FLAGS.batch_size, 
            token2id, is_training=False,
            total_max_len=total_max_len)

        dev_loss, dev_accuracy = 0, 0
        cnt = 0
        dev_label_lst, dev_true_lst = [], []
        for index, corpus in enumerate(dev_data):
            anchor, check, label = corpus
            Q = input_dict_formulation(anchor, check, label)
            try:
                [loss, logits, 
                pred_probs, accuracy] = model.infer(
                                    Q, mode="test",
                                    is_training=False, 
                                    symmetric=False)

                dev_label_lst += np.argmax(pred_probs, axis=-1).tolist()
                dev_true_lst += label.tolist()

                dev_loss += loss*anchor.shape[0]
                dev_accuracy += accuracy*anchor.shape[0]
                cnt += anchor.shape[0]
            except:
                continue
           
        dev_loss /= float(cnt)

        dev_accuracy = accuracy_score(dev_true_lst, dev_label_lst)
        dev_recall = recall_score(dev_true_lst, dev_label_lst, average='macro')
        dev_precision = precision_score(dev_true_lst, dev_label_lst, average='macro')
        dev_f1 = f1_score(dev_true_lst, dev_label_lst, average='macro')

        info["dev_loss"] = str(dev_loss)
        info["dev_accuracy"] = str(dev_accuracy)
        info["dev_f1"] = str(dev_f1)

        logger.info("epoch\t{}\tdev\tloss\t{}\taccuracy\t{}\tf1\t{}".format(epoch, dev_loss, 
                                                        dev_accuracy, dev_f1))

        if dev_f1 > best_dev_f1 or dev_loss < best_dev_loss:
            timestamp = str(int(time.time()))
            model.save_model(os.path.join(model_dir, model_name, "models"), model_name+"_{}_{}_{}".format(timestamp, dev_loss, dev_f1))
            best_dev_f1 = dev_f1
            best_dev_loss = dev_loss

            toleration_cnt = 0

            info["best_dev_loss"] = str(dev_loss)
            info["dev_f1"] = str(dev_f1)

            logger_utils.json_info(os.path.join(model_dir, model_name, "logs", "info.json"), info)
            logger.info("epoch\t{}\tbest_dev\tloss\t{}\tbest_accuracy\t{}\tbest_f1\t{}".format(epoch, dev_loss, 
                                                          dev_accuracy, best_dev_f1))
        else:
            toleration_cnt += 1
            if toleration_cnt == toleration:
                toleration_cnt = 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='model name')
    parser.add_argument('--model_config', type=str, help='model config path')
    parser.add_argument('--model_dir', type=str, help='model path')
    parser.add_argument('--config_prefix', type=str, help='config path')
    parser.add_argument('--gpu_id', type=str, help='gpu id')
    parser.add_argument('--train_path', type=str, help='train data path')
    parser.add_argument('--dev_path', type=str, help='dev data path')
    parser.add_argument('--w2v_path', type=str, help='pretrained w2v path')
    parser.add_argument('--vocab_path', type=str, help='vocab_path')
    parser.add_argument('--elmo_w2v_path', type=str, help='pretrained elmo w2v path')
    parser.add_argument('--user_dict_path', type=str, help='user_dict_path')
    parser.add_argument('--data_type', type=str, help='user_dict_path')


    args, unparsed = parser.parse_known_args()
    model_config = args.model_config

    with open(model_config, "r") as frobj:
        model_config = json.load(frobj)

    config = {}
    config["model_dir"] = args.model_dir
    config["model"] = args.model
    config["model_config_path"] = os.path.join(args.config_prefix, 
                            model_config.get(args.model, model_config["biblosa"]))
    config["gpu_id"] = args.gpu_id
    config["train_path"] = args.train_path
    config["w2v_path"] = args.w2v_path
    config["vocab_path"] = args.vocab_path
    config["dev_path"] = args.dev_path
    config["elmo_w2v_path"] = args.elmo_w2v_path
    config["data_type"] = args.data_type

    cut_tool = data_utils.cut_tool_api()
    try:
        cut_tool.init_config({
            "user_dict":args.user_dict_path})
        cut_tool.build_tool()
    except:
        cut_tool.init_config({
            })

        print("not existed dictionary path")
    
    train(config)


    










