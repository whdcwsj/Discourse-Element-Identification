import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import utils

import math
import copy

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, pos):
        gp_embds = Variable(self.pe[pos[:, :, 0]], requires_grad=False)
        lp_embds = Variable(self.pe[pos[:, :, 1]], requires_grad=False)
        pp_embds = Variable(self.pe[pos[:, :, 2]], requires_grad=False)
        return gp_embds, lp_embds, pp_embds

# 位置编码层
# 中文数据集：p_embd='add'，p_embd_dim=16
# 英文数据集：p_embd = None
class PositionLayer(nn.Module):
    def __init__(self, p_embd=None, p_embd_dim=16, zero_weight=False):
        super(PositionLayer, self).__init__()
        self.p_embd = p_embd
        self.p_embd_dim = p_embd_dim

        if zero_weight:
            self.pWeight = nn.Parameter(torch.zeros(3))
        else:
            self.pWeight = nn.Parameter(torch.ones(3))
        
        if p_embd == 'embd':
            self.g_embeddings = nn.Embedding(41, p_embd_dim)
            self.l_embeddings = nn.Embedding(21, p_embd_dim)
            self.p_embeddings = nn.Embedding(11, p_embd_dim)
        elif p_embd == 'embd_a':
            self.g_embeddings = nn.Embedding(100, p_embd_dim)
            self.l_embeddings = nn.Embedding(50, p_embd_dim)
            self.p_embeddings = nn.Embedding(30, p_embd_dim)
            self.gp_Linear = nn.Linear(p_embd_dim, 1)
            self.lp_Linear = nn.Linear(p_embd_dim, 1)
            self.pp_Linear = nn.Linear(p_embd_dim, 1)
        elif p_embd == 'embd_b':
            self.g_embeddings = nn.Embedding(41, p_embd_dim)
            self.l_embeddings = nn.Embedding(21, p_embd_dim)
            self.p_embeddings = nn.Embedding(11, p_embd_dim)
        elif p_embd == 'embd_c':
            self.pe = PositionalEncoding(p_embd_dim, 100)

    # sentpres: (batch_n, doc_l, hidden_dim*2)，pos: (batch_n,doc_l,6)
    # ['gpos', 'lpos', 'ppos', 'gid', 'lid', 'pid']
    def forward(self, sentpres, pos):
        # sentpres: (batch_n, doc_l, output_dim*2)
        if self.p_embd in utils.embd_name:
            pos = pos[:, :, 3:6].long()
        # 取前三个标签：['gpos', 'lpos', 'ppos']
        else:
            pos = pos[:, :, :3]
        if self.p_embd == 'embd':
            gp_embds = torch.tanh(self.g_embeddings(pos[:, :, 0]))
            lp_embds = torch.tanh(self.l_embeddings(pos[:, :, 1]))
            pp_embds = torch.tanh(self.p_embeddings(pos[:, :, 2]))
            sentpres = torch.cat((sentpres, gp_embds, lp_embds, pp_embds), dim=2)
        elif self.p_embd == 'embd_a':
            gp_embds = self.g_embeddings(pos[:, :, 0])
            lp_embds = self.l_embeddings(pos[:, :, 1])
            pp_embds = self.p_embeddings(pos[:, :, 2])
            sentpres = sentpres + self.pWeight[0] * torch.tanh(self.gp_Linear(gp_embds)) + \
                                  self.pWeight[1] * torch.tanh(self.lp_Linear(lp_embds)) + \
                                  self.pWeight[2] * torch.tanh(self.pp_Linear(pp_embds))
        elif self.p_embd == 'embd_b':
            gp_embds = self.g_embeddings(pos[:, :, 0])
            lp_embds = self.l_embeddings(pos[:, :, 1])
            pp_embds = self.p_embeddings(pos[:, :, 2])
            sentpres = sentpres + self.pWeight[0] * torch.tanh(gp_embds) + \
                                  self.pWeight[1] * torch.tanh(lp_embds) + \
                                  self.pWeight[2] * torch.tanh(pp_embds)
        elif self.p_embd == 'embd_c':
            gp_embds, lp_embds, pp_embds = self.pe(pos)
            sentpres = sentpres + self.pWeight[0] * gp_embds + \
                                  self.pWeight[1] * lp_embds + \
                                  self.pWeight[2] * pp_embds                   
        elif self.p_embd == 'cat':
            sentpres = torch.cat((sentpres, pos), dim=2)
        # 在zero_weight为False的情况下，pWeight初始化为1
        # 将前三个pos的数值1：1：1与sentence相加
        elif self.p_embd =='add':
            sentpres = sentpres + self.pWeight[0] * pos[:, :, :1] + self.pWeight[1] * pos[:, :, 1:2] + self.pWeight[2] * pos[:, :, 2:3]
        elif self.p_embd =='add1':
            sentpres = sentpres + self.pWeight[1] * pos[:, :, 1:2] + self.pWeight[2] * pos[:, :, 2:3]
        elif self.p_embd =='add2':
            sentpres = sentpres + self.pWeight[0] * pos[:, :, :1] + self.pWeight[2] * pos[:, :, 2:3]
        elif self.p_embd =='add3':
            sentpres = sentpres + self.pWeight[0] * pos[:, :, :1] + self.pWeight[1] * pos[:, :, 1:2]
        elif self.p_embd =='addg':
            sentpres = sentpres + self.pWeight[0] * pos[:, :, :1]
        elif self.p_embd =='addl':
            sentpres = sentpres + self.pWeight[1] * pos[:, :, 1:2]
        elif self.p_embd =='addp':
            sentpres = sentpres + self.pWeight[2] * pos[:, :, 2:3]
            
        return sentpres
        
    def init_embedding(self):
        gp_em_w = [[i/40] * self.p_embd_dim for i in range(41)]
        self.g_embeddings.weight = nn.Parameter(torch.FloatTensor(gp_em_w))
        lp_em_w = [[i/20] * self.p_embd_dim for i in range(21)]
        self.l_embeddings.weight = nn.Parameter(torch.FloatTensor(lp_em_w))
        pp_em_w = [[i/10] * self.p_embd_dim for i in range(11)]
        self.p_embeddings.weight = nn.Parameter(torch.FloatTensor(pp_em_w))
        
# 句子间注意力的自适应最大池化
# 使用了图像处理中的空间金字塔池化(SPP)
class InterSentenceSPPLayer(nn.Module):
    # self.hidden_dim*2, pool_type = 'max_pool'
    def __init__(self, hidden_dim, num_levels=4, pool_type='max_pool'):
        super(InterSentenceSPPLayer, self).__init__()
        self.linearK = nn.Linear(hidden_dim, hidden_dim)
        self.linearQ = nn.Linear(hidden_dim, hidden_dim)
        self.num_levels = num_levels
        self.pool_type = pool_type
        # 本文使用的四个bin：1，2，4，8
        # nn.ModuleList()将submodules保存在一个list中，可以索引
        # nn.ModuleList是无序性的序列，并且没有实现forward()方法

        # 自适应池化：nn.AdaptiveMaxPool1d() 只需要给定输出特征图的大小就好，其中通道数前后不发生变化
        if self.pool_type == 'max_pool':
            # nn.AdaptiveMaxPool1d()相当于分别输出最大的1，2，4，8号元素
            self.SPP = nn.ModuleList([nn.AdaptiveMaxPool1d(2**i) for i in range(num_levels)])
        else:
            # nn.AdaptiveAvgPool1d()具体计算方式存疑
            self.SPP = nn.ModuleList([nn.AdaptiveAvgPool1d(2**i) for i in range(num_levels)])

    # 输入一个句子embedding，过了一层LSTM和激活层之后的样子
    # sentpres: (batch_n, doc_l, hidden_dim*2)
    def forward(self, sentpres, is_softmax=False):
        # sentpres: (batch_n, doc_l, output_dim*2)
        doc_l = sentpres.size(1)
        key = self.linearK(sentpres)  # key/query:(batch,doc_l,output_dim*2)
        query = self.linearQ(sentpres)
        d_k = query.size(-1)
        features = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # features: (batch_n, doc_l, doc_l)

        # 自注意力层的
        if is_softmax:
            features = F.softmax(features, dim=2)
            # print(torch.sum(features, dim=2))

        features = torch.tanh(features)  # features:(batch,doc_l,doc_l)

        self.ft =  features
        pooling_layers = []
        for pooling in self.SPP:
            tensor = pooling(features)   # tensor:(batch,doc_l,SPP中的bin个数)
            pooling_layers.append(tensor)
            
        # print([x.size() for x in pooling_layers])
        self.features = torch.cat(pooling_layers, dim=-1)
        return self.features  
        

# 相比于InterSentenceSPPLayer，多了一个active_func
# self.pool_type中的多了一个max_pool与avg_pool的加和
# 其它其实差不多
class InterSentenceSPPLayer3(nn.Module):
    def __init__(self, hidden_dim, num_levels=4, pool_type='max_pool', active_func='tanh'):
        super(InterSentenceSPPLayer3, self).__init__()
        self.linearK = nn.Linear(hidden_dim, hidden_dim)
        self.linearQ = nn.Linear(hidden_dim, hidden_dim)
        self.num_levels = num_levels
        self.pool_type = pool_type
        if self.pool_type == 'max_pool':
            self.SPP = nn.ModuleList([nn.AdaptiveMaxPool1d(2**i) for i in range(num_levels)])
        elif self.pool_type == 'avg_pool':
            self.SPP = nn.ModuleList([nn.AdaptiveAvgPool1d(2**i) for i in range(num_levels)])
        else:
            self.SPP = nn.ModuleList([nn.AdaptiveAvgPool1d(2**i) for i in range(num_levels)] + [nn.AdaptiveMaxPool1d(2**i) for i in range(num_levels)])

        if active_func == 'tanh':
            self.active_func = nn.Tanh()
        elif active_func == 'relu':
            self.active_func = nn.ReLU()
        elif active_func == 'softmax':
            self.active_func = nn.Softmax(dim=2)
        else:
            self.active_func = None

    def forward(self, sentpres):
        # sentpres: (batch_n, doc_l, output_dim*2)
        doc_l = sentpres.size(1)
        key = self.linearK(sentpres)
        query = self.linearQ(sentpres)
        d_k = query.size(-1)
        features = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # features: (batch_n, doc_l, doc_l)

        # 先过一遍激活函数
        if self.active_func is not None:
            features = self.active_func(features)

        self.ft =  features
        pooling_layers = []
        for pooling in self.SPP:
            tensor = pooling(features)  # tensor:(batch,doc_l,SPP中的bin个数)
            pooling_layers.append(tensor)
            
        # print([x.size() for x in pooling_layers])
        self.features = torch.cat(pooling_layers, dim=-1)
        return self.features
