import os
import time
import torch


class Config:
    def __init__(self, name):
        # self.train_time = time.strftime('%m-%d_%H.%M', time.localtime())
        # self.num_layers = 1
        # self.gcn_layers = 3
        # self.dropout = 0.1

        self.human_model_name = name
        # batch_size = 20下，显存出现过不够的情况
        # Bert不训练的情况下
        # self.batch_size = 16
        # Bert训练的情况下
        self.batch_size = 10
        self.add_title = True
        self.lr = 0.2

        self.word_dim = 768
        self.hidden_dim = 128
        self.sent_dim = 128
        self.class_n = 8
        self.p_embd = 'add'
        self.p_embd_dim = 16
        self.pool_type = 'max_pool'
        self.bidirectional = True

        self.epoch = 300

        self.train_data_path = './data/new_Ch_train.json'
        self.dev_data_path = './data/new_Ch_dev.json'
        self.test_data_path = './data/Ch_test.json'

        self.model_save_path = './newmodel/cn/bert/'
        self.log_path = './newlog/cn/bert/'
        self.value_path = './newvalue/cn/bert/'

        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 路径前面加一个r,是为了保持路径在读取时不被漏读
        # 1、谷歌的Bert_chinese
        # self.bert_path = r'/home/wsj/bert_model/chinese/bert_chinese_L-12_H-768_A-12'
        # self.vocab_path = r'/home/wsj/bert_model/chinese/bert_chinese_L-12_H-768_A-12/vocab.txt'

        # 2、Chinese_bert_www
        self.bert_path = r'/home/wsj/bert_model/chinese/chinese_bert_wwm_pytorch'
        self.vocab_path = r'/home/wsj/bert_model/chinese/chinese_bert_wwm_pytorch/vocab.txt'

