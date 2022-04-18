import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import utils

import math
import copy

# 代码：https://blog.csdn.net/nocml/article/details/110920221
# transformer中的Positional Embedding位置编码
# 对于每个位置的PE是固定的，不会因为输入的句子不同而不同，且每个位置的PE大小为1∗n(n为word embedding的dim_size)
# transformer中使用正余弦波来计算PE
# d_model: 词嵌入维度，max_len: 每个句子的最大长度

# if p_embd in ['embd_b', 'embd_c']:
#     p_embd_dim = hidden_dim * 2 =128

class PositionalEncoding(nn.Module):
    # p_embd_dim=16, 100
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)  # pe:torch.Size([100, 16])
        # unsqueeze()这个函数主要是对数据维度进行扩充
        position = torch.arange(0., max_len).unsqueeze(1)  # position:torch.Size([max_len, 1])
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        # seq[start:end:step]
        # range(10)[::2]
        # [0, 2, 4, 6, 8]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 模型中需要保存下来的参数包括两种:
        # 一种是反向传播需要被optimizer更新的，称之为parameter
        # 一种是反向传播不需要被optimizer更新，称之为buffer
        # buffer我们把它认为是对模型效果有帮助的，但是却不是模型结构中超参数或者参数，不需要随着优化步骤进行更新的增益对象.
        # 注册之后我们就可以在模型保存后重加载时和模型结构与参数一同被加载.
        self.register_buffer('pe', pe)

    # (batch_n,doc_l,3)
    def forward(self, pos):
        # 把不需要更新参数的设置 requires_grad=False
        gp_embds = Variable(self.pe[pos[:, :, 0]], requires_grad=False)   # shape:(batch_n,doc_l,p_embd_dim)
        lp_embds = Variable(self.pe[pos[:, :, 1]], requires_grad=False)
        pp_embds = Variable(self.pe[pos[:, :, 2]], requires_grad=False)
        return gp_embds, lp_embds, pp_embds

# 位置编码层
# 中文数据集：p_embd='add'，p_embd_dim=16
# 英文数据集：p_embd = None，p_embd_dim=16
class PositionLayer(nn.Module):
    def __init__(self, p_embd=None, p_embd_dim=16, zero_weight=False, weight_matrix=None):
        super(PositionLayer, self).__init__()
        self.p_embd = p_embd
        self.p_embd_dim = p_embd_dim

        if zero_weight:
            self.pWeight = nn.Parameter(torch.zeros(3))
        else:
            if weight_matrix is None:
                self.pWeight = nn.Parameter(torch.ones(3))
            else:
                pWeight = torch.ones(3)
                pWeight[0] = pWeight[0] * weight_matrix[0]
                pWeight[1] = pWeight[1] * weight_matrix[1]
                pWeight[2] = pWeight[2] * weight_matrix[2]
                self.pWeight = nn.Parameter(pWeight)
        
        if p_embd == 'embd':
            # 第一个参数num_embedding词典的大小尺寸：，第二个参数embedding_dim：嵌入向量的维度
            # 输入： LongTensor (N, W), N = mini-batch, W = 每个mini-batch中提取的下标数
            # 输出： (N, W, embedding_dim)
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

        # if p_embd in ['embd_b', 'embd_c']:
        #     p_embd_dim = hidden_dim * 2 =128
        # 与'embd'相比较，p_embd_dim从16变换为128
        elif p_embd == 'embd_b':
            self.g_embeddings = nn.Embedding(41, p_embd_dim)
            self.l_embeddings = nn.Embedding(21, p_embd_dim)
            self.p_embeddings = nn.Embedding(11, p_embd_dim)
        # Transformer的位置编码方法
        elif p_embd == 'embd_c':
            self.pe = PositionalEncoding(p_embd_dim, 100)


    # sentpres: (batch_n, doc_l, hidden_dim*2)
    # pos: (batch_n,doc_l,6)
    # ['gpos', 'lpos', 'ppos', 'gid', 'lid', 'pid']
    def forward(self, sentpres, pos):
        # sentpres: (batch_n, doc_l, output_dim*2)
        # embd_name = ['embd', 'embd_a', 'embd_b', 'embd_c']
        if self.p_embd in utils.embd_name:
            pos = pos[:, :, 3:6].long()
        # 取前三个标签：['gpos', 'lpos', 'ppos']
        else:
            pos = pos[:, :, :3]

        # 这几个处理的是绝对位置
        # 后面三个直接torch.cat()
        if self.p_embd == 'embd':
            # pos[:, :, 0]  shape:(batch_n, doc_l)
            gp_embds = torch.tanh(self.g_embeddings(pos[:, :, 0]))   # shape:(batch_n, doc_l, p_embd_dim)
            lp_embds = torch.tanh(self.l_embeddings(pos[:, :, 1]))
            pp_embds = torch.tanh(self.p_embeddings(pos[:, :, 2]))
            sentpres = torch.cat((sentpres, gp_embds, lp_embds, pp_embds), dim=2)
        # 后面三个先MLP，再加和
        elif self.p_embd == 'embd_a':
            gp_embds = self.g_embeddings(pos[:, :, 0])
            lp_embds = self.l_embeddings(pos[:, :, 1])
            pp_embds = self.p_embeddings(pos[:, :, 2])
            sentpres = sentpres + self.pWeight[0] * torch.tanh(self.gp_Linear(gp_embds)) + \
                                  self.pWeight[1] * torch.tanh(self.lp_Linear(lp_embds)) + \
                                  self.pWeight[2] * torch.tanh(self.pp_Linear(pp_embds))
        # 后面三个先tanh，再加和
        elif self.p_embd == 'embd_b':
            gp_embds = self.g_embeddings(pos[:, :, 0])
            lp_embds = self.l_embeddings(pos[:, :, 1])
            pp_embds = self.p_embeddings(pos[:, :, 2])
            sentpres = sentpres + self.pWeight[0] * torch.tanh(gp_embds) + \
                                  self.pWeight[1] * torch.tanh(lp_embds) + \
                                  self.pWeight[2] * torch.tanh(pp_embds)

        # 采用transformer中的位置编码
        elif self.p_embd == 'embd_c':
            gp_embds, lp_embds, pp_embds = self.pe(pos)
            sentpres = sentpres + self.pWeight[0] * gp_embds + \
                                  self.pWeight[1] * lp_embds + \
                                  self.pWeight[2] * pp_embds                   

        # 下面处理的是相对位置
        elif self.p_embd == 'cat':
            sentpres = torch.cat((sentpres, pos), dim=2)
        # 在zero_weight为False的情况下，pWeight初始化为1
        # 将前三个pos的数值1：1：1与sentence相加
        elif self.p_embd =='add':
            sentpres = sentpres + self.pWeight[0] * pos[:, :, :1] + self.pWeight[1] * pos[:, :, 1:2] + self.pWeight[2] * pos[:, :, 2:3]
        # 跳过第一个相对位置
        elif self.p_embd =='add1':
            sentpres = sentpres + self.pWeight[1] * pos[:, :, 1:2] + self.pWeight[2] * pos[:, :, 2:3]
        # 跳过第二个相对位置
        elif self.p_embd =='add2':
            sentpres = sentpres + self.pWeight[0] * pos[:, :, :1] + self.pWeight[2] * pos[:, :, 2:3]
        # 跳过第三个相对位置
        elif self.p_embd =='add3':
            sentpres = sentpres + self.pWeight[0] * pos[:, :, :1] + self.pWeight[1] * pos[:, :, 1:2]
        # 只留第一个相对位置
        elif self.p_embd =='addg':
            sentpres = sentpres + self.pWeight[0] * pos[:, :, :1]
        # 只留第二个相对位置
        elif self.p_embd =='addl':
            sentpres = sentpres + self.pWeight[1] * pos[:, :, 1:2]
        # 只留第三个相对位置
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
