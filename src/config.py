import os
import time
import torch

class Config:
    def __init__(self):
        # self.train_time = time.strftime('%m-%d_%H.%M', time.localtime())
        # self.num_layers = 1
        # self.hidden_dim = 128
        # self.bidirectional = True
        # self.embedding_dim = 768
        # self.dropout = 0.1
        # self.epoch = 700
        # self.gcn_layers = 3

        self.train_data_path = '../data/new_Ch_train.json'
        self.dev_data_path = '../data/new_Ch_dev.json'
        self.test_data_path = '../data/Ch_test.json'

        self.model_name = 'wsj_bert_test'
        self.model_save_path = '../newmodel/cn/bert/'
        self.log_path = '../newlog/cn/bert/'
        self.value_path = '../newvalue/cn/bert/'

        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 路径前面加一个r,是为了保持路径在读取时不被漏读
        self.bert_path = r'/home/wsj/bert_model/chinese/bert_chinese_L-12_H-768_A-12'
        self.vocab_path = r'/home/wsj/bert_model/chinese/bert_chinese_L-12_H-768_A-12/vocab.txt'


        if not os.path.exists(self.model_save_path + self.model_name):
            os.mkdir(self.model_save_path + self.model_name)
        if not os.path.exists(self.log_path + self.model_name):
            os.mkdir(self.log_path + self.model_name)
        if not os.path.exists(self.value_path + self.model_name):
            os.mkdir(self.value_path + self.model_name)