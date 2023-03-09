# encoding: utf-8
# @author: 王宇静轩
# @file: picture_visualization.py
# @time: 2023/3/9 上午9:45


import utils
import os
import torch
import numpy as np
import newtrain
import chinese_train
import newtrain_en
import newtrain_en_ft
import argparse
import utils_e
from transformers import BertTokenizer


def Chinese_test(model_dir):
    in_file = './data/Ch_test.json'
    embed_filename = './embd/new_embeddings2.txt'
    title = True

    print('load Chinense Embeddings...')
    max_len = 40

    en_documents, en_labels, features, vec_size = utils.getSamplesAndFeatures(in_file, embed_filename, title=title)
    pad_documents, pad_labels, essay_length = utils.sentence_padding_dgl(en_documents, en_labels, max_len, vec_size)
    is_mask = False

    from chinese_train import test_dgl

    tag_model = torch.load(model_dir, map_location='cpu')

    temp_accurancy, loss, a = test_dgl(tag_model, pad_documents, pad_labels, features, essay_length, 'cpu', batch_n=1,
                                       title=title, is_mask=is_mask)

    return temp_accurancy


def English_test(model_dir):
    in_file = './data/En_test.json'
    title = True
    is_word = False
    max_len = 40

    embeddings = torch.load('./embd/bert-base-uncased-word_embeddings.pkl')

    BERT_PATH = './bert/bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(BERT_PATH)

    en_documents, en_labels, features = utils_e.getEnglishSamplesBertId(in_file, tokenizer, title=title,
                                                                        is_word=is_word)
    pad_documents, pad_labels, essay_length = utils_e.sentencePaddingId_dgl(en_documents, en_labels, max_len)

    from newtrain_en import test_dgl

    tag_model = torch.load(model_dir, map_location='cpu')

    temp_accurancy, loss, a = test_dgl(tag_model, pad_documents, pad_labels, features, essay_length, 'cpu', batch_n=1,
                                       title=title,
                                       embeddings=embeddings)

    return temp_accurancy


if __name__ == "__main__":
    # 调用这个目录下的所有模型进行test
    # 0是中文，1是英文

    parser = argparse.ArgumentParser(description='Test Discourse', usage='newtest.py [<args>] [-h | --help]')
    parser.add_argument('--type_id', default=0, type=int, help='Set type num.')
    args = parser.parse_args()
    test_type_id = args.type_id

    # useful_information
    # best_model_path
    # cn_best
    # /home/wsj/dgl_file/cn_best_result/SAGE部分连通/model/dgl_st_rs_sppm_128_128_ap-01-05_14.29_seed_300/dgl_st_rs_sppm_128_128_ap_t_260_best.pk
    # en_best
    # /home/wsj/dgl_file/en_best_result/model/dgl1_p4_lstm_w3_size1/dgl_st_rs_sppm_64_64_ap-03-04_16.13_seed_300/e_dgl_st_rs_sppm_64_64_ap_t_0_best.pk

    if test_type_id == 0:
        model_dir = "/home/wsj/dgl_file/cn_best_result/SAGE部分连通/model/dgl_st_rs_sppm_128_128_ap-01-05_14.29_seed_300/dgl_st_rs_sppm_128_128_ap_t_260_best.pk"
        acc = Chinese_test(model_dir)
        print(acc)

    elif test_type_id == 1:
        model_dir = "/home/wsj/dgl_file/en_best_result/model/dgl1_p4_lstm_w3_size1/dgl_st_rs_sppm_64_64_ap-03-04_16.13_seed_300/e_dgl_st_rs_sppm_64_64_ap_t_0_best.pk"
        acc = English_test(model_dir)
        print(acc)
