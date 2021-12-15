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

        self.work_path = './newmodel/cn/'

        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.bert_path = r'/home/wsj/bert_model/chinese/bert_chinese_L-12_H-768_A-12'
        self.vocab_path = r'/home/wsj/bert_model/chinese/bert_chinese_L-12_H-768_A-12/vocab.txt'




        # self.work_path = './src/result/GCN_elmo/' + self.model_name + '/seed-' + str(self.seed)
        self.model_save_path = self.work_path + '/model'
        self.log_path = self.work_path + '/logs'
        if not os.path.isdir(self.model_save_path):
            os.makedirs(self.model_save_path)
        if not os.path.isdir(self.log_path):
            os.makedirs(self.log_path)

        self.gcn_layers = 3