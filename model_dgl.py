import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dgltor
import numpy as np
import pandas as pd

from subLayer import *

# pytorch里面自定义层也是通过继承自nn.Module类来实现的
# pytorch里面一般是没有层的概念，层也是当成一个模型来处理的

# 针对中文数据集，POS2(先加位置信息，再加DGL)
class STWithRSbySPP_DGL(nn.Module):
    def __init__(self, word_dim, hidden_dim, sent_dim, class_n, p_embd=None, pos_dim=0, p_embd_dim=16,
                 pool_type='max_pool', dgl_layer=1):
        # p_embd: 'cat', 'add','embd', 'embd_a'
        super(STWithRSbySPP_DGL, self).__init__()
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.sent_dim = sent_dim
        self.class_n = class_n
        self.p_embd = p_embd
        self.p_embd_dim = p_embd_dim
        self.pool_type = pool_type

        self.dropout = nn.Dropout(0.1)
        self.sentLayer = nn.LSTM(self.word_dim, self.hidden_dim, bidirectional=True)

        # 为什么sent_dim*2+30？？？因为要接两个句间注意力
        self.classifier = nn.Linear(self.sent_dim * 2 + 30, self.class_n)
        # 配合avg与max加和时进行使用
        # self.classifier = nn.Linear(self.sent_dim * 2 + 60, self.class_n)

        # 单独dgl，不加SPP
        # self.classifier = nn.Linear(self.sent_dim * 2, self.class_n)

        self.posLayer = PositionLayer(p_embd, p_embd_dim)

        self.sfLayer = InterSentenceSPPLayer(self.hidden_dim * 2, pool_type=self.pool_type)
        self.rfLayer = InterSentenceSPPLayer(self.hidden_dim * 2, pool_type=self.pool_type)

        # 配合avg与max加和时进行使用
        # self.sfLayer = InterSentenceSPPLayer3(self.hidden_dim*2, pool_type = self.pool_type)
        # self.rfLayer = InterSentenceSPPLayer3(self.hidden_dim*2, pool_type = self.pool_type)

        # 这个是将三个位置编码torch.cat到了一起，所以input_size要再加上 p_embd_dim*3
        if p_embd == 'embd':
            self.tagLayer = nn.LSTM(self.hidden_dim * 2 + p_embd_dim * 3, self.sent_dim, bidirectional=True)
        # 这个是将三个位置编码torch.cat到了一起，所以input_size要再加上3
        elif p_embd == 'cat':
            self.tagLayer = nn.LSTM(self.hidden_dim * 2 + 3, self.sent_dim, bidirectional=True)
        else:
            self.tagLayer = nn.LSTM(self.hidden_dim * 2, self.sent_dim, bidirectional=True)

        # 是否添加norm，后续需要进行尝试:'right'或者'none',default='both'
        self.SAGE_GCN = nn.ModuleList(
            [dgltor.SAGEConv(self.sent_dim * 2, self.sent_dim * 2, aggregator_type='gcn', feat_drop=0.1, bias=True, activation=nn.ReLU())
             for _ in range(dgl_layer)])

        self.transition_layer = nn.Sequential(
            nn.Linear(self.sent_dim * 2 * dgl_layer , self.sent_dim * 2),
            nn.Dropout(0.1)
        )

    # batch_size，一篇文章的句子个数
    def init_hidden(self, batch_n, doc_l, device='cpu'):

        self.sent_hidden = (torch.rand(2, batch_n * doc_l, self.hidden_dim, device=device).uniform_(-0.01, 0.01),
                            torch.rand(2, batch_n * doc_l, self.hidden_dim, device=device).uniform_(-0.01, 0.01))
        # 两个元组，shape都是(2,batch_n,hidden_dim)
        self.tag_hidden = (torch.rand(2, batch_n, self.sent_dim, device=device).uniform_(-0.01, 0.01),
                           torch.rand(2, batch_n, self.sent_dim, device=device).uniform_(-0.01, 0.01))

    #  sentence_encoding:(doc_l, sent_dim*2)，actual_length:实际节点个数
    def build_graph(self, sentence_encoding, actual_length, device='cpu'):

        nodes_num = actual_length
        # 保存每对边的连接
        edges = []
        # 保存边的节点权重
        edges_weight = []

        for i in range(nodes_num):
            for j in range(nodes_num):
                # 构建双向图(自循环已经考虑进去了)
                edges.append((i, j))
                # 计算余弦相似度
                weight = torch.cosine_similarity(sentence_encoding[i], sentence_encoding[j], dim=0)
                edges_weight.append(weight)

                # whether to add another self-loop，这条边有ferature(1.)
                # if i == j:
                #     edges.append((i, j))
                #     edges_weight.append(weight)

        # 必须要先张量化
        edges = torch.tensor(edges)
        # 若超出实际长度则产生孤立节点
        graph = dgl.graph((edges[:, 0], edges[:, 1]), num_nodes=len(sentence_encoding)).to(device)

        # whether to add another self-loop，此时edges的feature为0
        # graph = dgl.add_self_loop(graph)

        edges_weight = torch.tensor(edges_weight).to(device)
        return graph, edges_weight


    # 测试集情况
    # document:(batch_n,doc_l,40,200)
    # pos:(batch_n,doc_l,6)  6个特征：['gpos', 'lpos', 'ppos', 'gid', 'lid', 'pid']
    # mask:NONE
    def forward(self, documents, pos=None, length_essay=None, device='cpu', mask=None):
        batch_n, doc_l, sen_l, _ = documents.size()  # documents: (batch_n, doc_l, sen_l, word_dim)
        self.init_hidden(batch_n=batch_n, doc_l=doc_l, device=device)
        documents = documents.view(batch_n * doc_l, sen_l, -1).transpose(0,1)  # documents: (sen_l, batch_n*doc_l, word_dim)

        sent_out, _ = self.sentLayer(documents, self.sent_hidden)  # sent_out: (sen_l, batch_n*doc_l, hidden_dim*2)
        sent_out = self.dropout(sent_out)

        if mask is None:
            # sentpres：(batch_n*doc_l,1,256)
            # 激活函数
            sentpres = torch.tanh(torch.mean(sent_out, dim=0))  # sentpres: (batch_n*doc_l, hidden_dim*2)
        else:
            sent_out = sent_out.masked_fill(mask.transpose(1, 0).unsqueeze(-1).expand_as(sent_out), 0)
            sentpres = torch.tanh(torch.sum(sent_out, dim=0) / (sen_l - mask.sum(dim=1).float() + 1e-9).unsqueeze(-1))

        sentpres = sentpres.view(batch_n, doc_l, self.hidden_dim * 2)  # sentpres: (batch_n, doc_l, hidden_dim*2)

        # sentence embedding的句间注意力
        sentFt = self.sfLayer(sentpres)  # sentFt:(batch_n, doc_l,15)
        # sentFt = self.dropout(sentFt)

        # "add"情况下，将前三个pos位置1：1：1与sentence加和; ['gpos', 'lpos', 'ppos']
        sentpres = self.posLayer(sentpres, pos)  # sentpres:(batch_n, doc_l, hidden_dim*2)


        # 可以加dgl的位置1(先加dgl，再加pos)


        sentpres = sentpres.transpose(0, 1)  # sentpres: (doc_l, batch_n, hidden_dim*2)

        tag_out, _ = self.tagLayer(sentpres, self.tag_hidden)  # tag_out: (doc_l, batch_n, sent_dim*2)
        tag_out = self.dropout(tag_out)

        tag_out = torch.tanh(tag_out)

        tag_out = tag_out.transpose(0, 1)  # tag_out: (batch_n, doc_l, sent_dim*2)


        # 可以加dgl的位置2(先加pos，再加dgl)
        # ---------------------------------------------------------------------

        # 加入GCN
        # length_essay
        # tensor([30, 30, 30, 30, 30, 30, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31,
        #         31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31,
        #         31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31],
        #        device='cuda:0')

        # 存储每篇文章经过dgl之后的feature
        total_sentence_feature = []

        for i in range(len(length_essay)):
            inner_sentennce = tag_out[i]
            node_length = length_essay[i]
            # build graph
            graph, edge_weight = self.build_graph(inner_sentennce, node_length, device=device)

            current_essay_sentence = []
            # try add different GCN，句间交互
            for sage_gcn in self.SAGE_GCN:
                # 输出：(node_nums, self.sent_dim * 2)
                inner_sentennce = sage_gcn(graph, inner_sentennce, edge_weight)
                current_essay_sentence.append(inner_sentennce)
            # 每篇文章的
            current_essay_sentence = torch.cat(current_essay_sentence, dim=-1)  # current_essay_sentence:(node_nums, self.sent_dim * 2 * dgl_layer)
            # 统一一下维度
            current_essay_sentence = self.transition_layer(current_essay_sentence)  # current_essay_sentence:(node_nums, self.sent_dim * 2)
            # 加入总体中
            total_sentence_feature.append(current_essay_sentence)

        # 一个batch下的所有文章
        dgl_out = torch.stack(total_sentence_feature, dim=0)  # total_sentence_feature:(batch_size, node_nums, self.sent_dim * 2)


        # ----------------------------------------------------------------------


        roleFt = self.rfLayer(tag_out)  # roleFt:(batch_n, doc_l, 15)
        # roleFt = self.dropout(roleFt)

        new_out = torch.cat((dgl_out, sentFt, roleFt), dim=2)  # tag_out: (batch_n, doc_l, sent_dim*2+30)  (1,8,286)
        # new_out = self.dropout(tag_out)

        # class_n用在了这里
        result = self.classifier(new_out)  # tag_out: (batch_n, doc_l, class_n)

        # log_softmax 和 nll_loss配套使用
        result = F.log_softmax(result, dim=2)  # result: (batch_n, doc_l, class_n)
        return result

    def getModelName(self):
        # pool_type='max_pool'的第一个字母m
        name = 'dgl_st_rs_spp%s' % self.pool_type[0]
        name += '_' + str(self.hidden_dim) + '_' + str(self.sent_dim)
        if self.p_embd == 'cat':
            name += '_cp'
        elif self.p_embd == 'add':
            name += '_ap'
        elif self.p_embd == 'embd':
            name += '_em'
        elif self.p_embd == 'embd_a':
            name += '_em_a'
        elif self.p_embd:
            name += '_' + self.p_embd
        return name



# 将DGL放在pos1位置
# 右边的content self attention仍采用原始的sentence_embeeding
# 左边和右边的采用过了DGL之后的sentence_embeeding
class STWithRSbySPP_DGL_POS1(nn.Module):
    def __init__(self, word_dim, hidden_dim, sent_dim, class_n, p_embd=None, pos_dim=0, p_embd_dim=16,
                 pool_type='max_pool', dgl_layer=1, gcn_aggr='gcn', weight_id=1, loop=0):
        # p_embd: 'cat', 'add','embd', 'embd_a'
        super(STWithRSbySPP_DGL_POS1, self).__init__()
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.sent_dim = sent_dim
        self.class_n = class_n
        self.p_embd = p_embd
        self.p_embd_dim = p_embd_dim
        self.pool_type = pool_type

        self.dropout = nn.Dropout(0.1)
        self.sentLayer = nn.LSTM(self.word_dim, self.hidden_dim, bidirectional=True)

        # 为什么sent_dim*2+30？？？因为要接两个句间注意力
        self.classifier = nn.Linear(self.sent_dim * 2 + 30, self.class_n)
        # 配合avg与max加和时进行使用
        # self.classifier = nn.Linear(self.sent_dim * 2 + 60, self.class_n)

        # 单独dgl，不加SPP
        # self.classifier = nn.Linear(self.sent_dim * 2, self.class_n)

        self.posLayer = PositionLayer(p_embd, p_embd_dim)

        self.sfLayer = InterSentenceSPPLayer(self.hidden_dim * 2, pool_type=self.pool_type)
        self.rfLayer = InterSentenceSPPLayer(self.hidden_dim * 2, pool_type=self.pool_type)

        # 配合avg与max加和时进行使用
        # self.sfLayer = InterSentenceSPPLayer3(self.hidden_dim*2, pool_type = self.pool_type)
        # self.rfLayer = InterSentenceSPPLayer3(self.hidden_dim*2, pool_type = self.pool_type)

        # 这个是将三个位置编码torch.cat到了一起，所以input_size要再加上 p_embd_dim*3
        if p_embd == 'embd':
            self.tagLayer = nn.LSTM(self.hidden_dim * 2 + p_embd_dim * 3, self.sent_dim, bidirectional=True)
        # 这个是将三个位置编码torch.cat到了一起，所以input_size要再加上3
        elif p_embd == 'cat':
            self.tagLayer = nn.LSTM(self.hidden_dim * 2 + 3, self.sent_dim, bidirectional=True)
        else:
            self.tagLayer = nn.LSTM(self.hidden_dim * 2, self.sent_dim, bidirectional=True)

        self.edge_weight_id = weight_id
        self.gcn_loop = loop

        # 是否添加norm，后续需要进行尝试:'right'或者'none',default='both'
        # gcn 聚合可以理解为周围所有的邻居结合和当前节点的均值
        self.SAGE_GCN = nn.ModuleList(
            [dgltor.SAGEConv(self.sent_dim * 2, self.sent_dim * 2, aggregator_type=gcn_aggr, feat_drop=0.1, bias=True, activation=nn.ReLU())
             for _ in range(dgl_layer)])

        self.transition_layer = nn.Sequential(
            nn.Linear(self.sent_dim * 2 * (dgl_layer + 1), self.sent_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    # batch_size，一篇文章的句子个数
    def init_hidden(self, batch_n, doc_l, device='cpu'):

        self.sent_hidden = (torch.rand(2, batch_n * doc_l, self.hidden_dim, device=device).uniform_(-0.01, 0.01),
                            torch.rand(2, batch_n * doc_l, self.hidden_dim, device=device).uniform_(-0.01, 0.01))
        # 两个元组，shape都是(2,batch_n,hidden_dim)
        self.tag_hidden = (torch.rand(2, batch_n, self.sent_dim, device=device).uniform_(-0.01, 0.01),
                           torch.rand(2, batch_n, self.sent_dim, device=device).uniform_(-0.01, 0.01))

    #  sentence_encoding:(doc_l, sent_dim*2)，actual_length:实际节点个数
    def build_graph(self, sentence_encoding, actual_length, device='cpu'):

        nodes_num = actual_length
        # 保存每对边的连接
        edges = []
        # 保存边的节点权重
        edges_weight = []

        for i in range(nodes_num):
            for j in range(nodes_num):
                # 构建双向图(自循环已经考虑进去了)
                edges.append((i, j))

                # 计算余弦相似度
                if self.edge_weight_id == 1:
                    weight = torch.cosine_similarity(sentence_encoding[i], sentence_encoding[j], dim=0)
                    edges_weight.append(weight)
                # Pearson相似度
                elif self.edge_weight_id == 2:
                    pearson = np.corrcoef(sentence_encoding[i].cpu().detach().numpy(), sentence_encoding[j].cpu().detach().numpy())[0][1]
                    weight = torch.tensor(pearson, dtype=torch.float).to(device)
                    edges_weight.append(weight)
                # 欧氏距离，1/分母作为权重
                elif self.edge_weight_id == 3:
                    distance = torch.pairwise_distance(sentence_encoding[i][None, :], sentence_encoding[j][None, :])
                    weight = 1 / (1 + distance[0])
                    edges_weight.append(weight)
                # kendall系数
                elif self.edge_weight_id == 4:
                    kendall = pd.Series(sentence_encoding[i].cpu().detach().numpy()).corr(
                        pd.Series(sentence_encoding[j].cpu().detach().numpy()), method="kendall")
                    weight = torch.tensor(kendall, dtype=torch.float).to(device)
                    edges_weight.append(weight)

                # whether to add another self-loop，这条边有ferature(1.)
                if self.gcn_loop:
                    if i == j:
                        edges.append((i, j))
                        edges_weight.append(weight)

        # 必须要先张量化
        edges = torch.tensor(edges)
        # 若超出实际长度则产生孤立节点
        graph = dgl.graph((edges[:, 0], edges[:, 1]), num_nodes=len(sentence_encoding)).to(device)

        # whether to add another self-loop，此时edges的feature为0
        # graph = dgl.add_self_loop(graph)

        edges_weight = torch.tensor(edges_weight).to(device)
        return graph, edges_weight


    # 测试集情况
    # document:(batch_n,doc_l,40,200)
    # pos:(batch_n,doc_l,6)  6个特征：['gpos', 'lpos', 'ppos', 'gid', 'lid', 'pid']
    # mask:NONE
    def forward(self, documents, pos=None, length_essay=None, device='cpu', mask=None):
        batch_n, doc_l, sen_l, _ = documents.size()  # documents: (batch_n, doc_l, sen_l, word_dim)
        self.init_hidden(batch_n=batch_n, doc_l=doc_l, device=device)
        documents = documents.view(batch_n * doc_l, sen_l, -1).transpose(0,1)  # documents: (sen_l, batch_n*doc_l, word_dim)

        sent_out, _ = self.sentLayer(documents, self.sent_hidden)  # sent_out: (sen_l, batch_n*doc_l, hidden_dim*2)
        sent_out = self.dropout(sent_out)

        if mask is None:
            # sentpres：(batch_n*doc_l,1,256)
            # 激活函数
            sentpres = torch.tanh(torch.mean(sent_out, dim=0))  # sentpres: (batch_n*doc_l, hidden_dim*2)
        else:
            sent_out = sent_out.masked_fill(mask.transpose(1, 0).unsqueeze(-1).expand_as(sent_out), 0)
            sentpres = torch.tanh(torch.sum(sent_out, dim=0) / (sen_l - mask.sum(dim=1).float() + 1e-9).unsqueeze(-1))

        sentpres = sentpres.view(batch_n, doc_l, self.hidden_dim * 2)  # sentpres: (batch_n, doc_l, hidden_dim*2)

        # sentence embedding的句间注意力
        sentFt = self.sfLayer(sentpres)  # sentFt:(batch_n, doc_l,15)
        # sentFt = self.dropout(sentFt)


        # 可以加dgl的位置1(先加dgl，再加pos)
        # ---------------------------------------------------------------------

        # 加入GCN
        # length_essay
        # tensor([30, 30, 30, 30, 30, 30, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31,
        #         31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31,
        #         31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31],
        #        device='cuda:0')

        # 存储每篇文章经过dgl之后的feature
        total_sentence_feature = []

        for i in range(len(length_essay)):
            inner_sentennce = sentpres[i]
            node_length = length_essay[i]
            # build graph
            graph, edge_weight = self.build_graph(inner_sentennce, node_length, device=device)

            current_essay_sentence = [inner_sentennce]
            # try add different GCN，句间交互
            for sage_gcn in self.SAGE_GCN:
                # 输出：(node_nums, self.hidden_dim * 2)
                inner_sentennce = sage_gcn(graph, inner_sentennce, edge_weight)
                current_essay_sentence.append(inner_sentennce)
            # 每篇文章的
            current_essay_sentence = torch.cat(current_essay_sentence, dim=-1)  # current_essay_sentence:(node_nums, self.hidden_dim * 2 * dgl_layer)
            # 统一一下维度
            current_essay_sentence = self.transition_layer(current_essay_sentence)  # current_essay_sentence:(node_nums, self.hidden_dim * 2)
            # 加入总体中
            total_sentence_feature.append(current_essay_sentence)

        # 一个batch下的所有文章
        dgl_out = torch.stack(total_sentence_feature, dim=0)  # total_sentence_feature:(batch_size, node_nums, self.hidden_dim * 2)

        # ----------------------------------------------------------------------

        # "add"情况下，将前三个pos位置1：1：1与sentence加和; ['gpos', 'lpos', 'ppos']
        sentpres = self.posLayer(dgl_out, pos)  # sentpres:(batch_n, doc_l, hidden_dim*2)

        sentpres = sentpres.transpose(0, 1)  # sentpres: (doc_l, batch_n, hidden_dim*2)

        tag_out, _ = self.tagLayer(sentpres, self.tag_hidden)  # tag_out: (doc_l, batch_n, sent_dim*2)
        tag_out = self.dropout(tag_out)

        tag_out = torch.tanh(tag_out)

        tag_out = tag_out.transpose(0, 1)  # tag_out: (batch_n, doc_l, sent_dim*2)


        # 可以加dgl的位置2(先加pos，再加dgl)



        roleFt = self.rfLayer(tag_out)  # roleFt:(batch_n, doc_l, 15)
        # roleFt = self.dropout(roleFt)

        new_out = torch.cat((tag_out, sentFt, roleFt), dim=2)  # tag_out: (batch_n, doc_l, sent_dim*2+30)  (1,8,286)
        # new_out = self.dropout(tag_out)

        # class_n用在了这里
        result = self.classifier(new_out)  # tag_out: (batch_n, doc_l, class_n)

        # log_softmax 和 nll_loss配套使用
        result = F.log_softmax(result, dim=2)  # result: (batch_n, doc_l, class_n)
        return result

    def getModelName(self):
        # pool_type='max_pool'的第一个字母m
        name = 'dgl_st_rs_spp%s' % self.pool_type[0]
        name += '_' + str(self.hidden_dim) + '_' + str(self.sent_dim)
        if self.p_embd == 'cat':
            name += '_cp'
        elif self.p_embd == 'add':
            name += '_ap'
        elif self.p_embd == 'embd':
            name += '_em'
        elif self.p_embd == 'embd_a':
            name += '_em_a'
        elif self.p_embd:
            name += '_' + self.p_embd
        return name



# 将DGL放在pos3位置(将DGL放在最下层的位置)
# 对原始的sentence_embeeding先进行DGL，剩下的三部分均在此基础上进行
class STWithRSbySPP_DGL_POS_Bottom(nn.Module):
    def __init__(self, word_dim, hidden_dim, sent_dim, class_n, p_embd=None, pos_dim=0, p_embd_dim=16,
                 pool_type='max_pool', dgl_layer=1, gcn_aggr='gcn', weight_id=1, loop=0):
        # p_embd: 'cat', 'add','embd', 'embd_a'
        super(STWithRSbySPP_DGL_POS_Bottom, self).__init__()
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.sent_dim = sent_dim
        self.class_n = class_n
        self.p_embd = p_embd
        self.p_embd_dim = p_embd_dim
        self.pool_type = pool_type

        self.dropout = nn.Dropout(0.1)
        self.sentLayer = nn.LSTM(self.word_dim, self.hidden_dim, bidirectional=True)

        # 为什么sent_dim*2+30？？？因为要接两个句间注意力
        self.classifier = nn.Linear(self.sent_dim * 2 + 30, self.class_n)
        # 配合avg与max加和时进行使用
        # self.classifier = nn.Linear(self.sent_dim * 2 + 60, self.class_n)

        # 单独dgl，不加SPP
        # self.classifier = nn.Linear(self.sent_dim * 2, self.class_n)

        self.posLayer = PositionLayer(p_embd, p_embd_dim)

        self.sfLayer = InterSentenceSPPLayer(self.hidden_dim * 2, pool_type=self.pool_type)
        self.rfLayer = InterSentenceSPPLayer(self.hidden_dim * 2, pool_type=self.pool_type)

        # 配合avg与max加和时进行使用
        # self.sfLayer = InterSentenceSPPLayer3(self.hidden_dim*2, pool_type = self.pool_type)
        # self.rfLayer = InterSentenceSPPLayer3(self.hidden_dim*2, pool_type = self.pool_type)

        # 这个是将三个位置编码torch.cat到了一起，所以input_size要再加上 p_embd_dim*3
        if p_embd == 'embd':
            self.tagLayer = nn.LSTM(self.hidden_dim * 2 + p_embd_dim * 3, self.sent_dim, bidirectional=True)
        # 这个是将三个位置编码torch.cat到了一起，所以input_size要再加上3
        elif p_embd == 'cat':
            self.tagLayer = nn.LSTM(self.hidden_dim * 2 + 3, self.sent_dim, bidirectional=True)
        else:
            self.tagLayer = nn.LSTM(self.hidden_dim * 2, self.sent_dim, bidirectional=True)

        self.edge_weight_id = weight_id
        self.gcn_loop = loop

        # 是否添加norm，后续需要进行尝试:'right'或者'none',default='both'
        # gcn 聚合可以理解为周围所有的邻居结合和当前节点的均值
        self.SAGE_GCN = nn.ModuleList(
            [dgltor.SAGEConv(self.sent_dim * 2, self.sent_dim * 2, aggregator_type=gcn_aggr, feat_drop=0.1, bias=True, activation=nn.ReLU())
             for _ in range(dgl_layer)])

        self.transition_layer = nn.Sequential(
            nn.Linear(self.sent_dim * 2 * (dgl_layer + 1), self.sent_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    # batch_size，一篇文章的句子个数
    def init_hidden(self, batch_n, doc_l, device='cpu'):

        self.sent_hidden = (torch.rand(2, batch_n * doc_l, self.hidden_dim, device=device).uniform_(-0.01, 0.01),
                            torch.rand(2, batch_n * doc_l, self.hidden_dim, device=device).uniform_(-0.01, 0.01))
        # 两个元组，shape都是(2,batch_n,hidden_dim)
        self.tag_hidden = (torch.rand(2, batch_n, self.sent_dim, device=device).uniform_(-0.01, 0.01),
                           torch.rand(2, batch_n, self.sent_dim, device=device).uniform_(-0.01, 0.01))

    #  sentence_encoding:(doc_l, sent_dim*2)，actual_length:实际节点个数
    def build_graph(self, sentence_encoding, actual_length, device='cpu'):

        nodes_num = actual_length
        # 保存每对边的连接
        edges = []
        # 保存边的节点权重
        edges_weight = []

        for i in range(nodes_num):
            for j in range(nodes_num):
                # 构建双向图(自循环已经考虑进去了)
                edges.append((i, j))
                weight = 0

                # 计算余弦相似度
                if self.edge_weight_id == 1:
                    weight = torch.cosine_similarity(sentence_encoding[i], sentence_encoding[j], dim=0)
                    edges_weight.append(weight)
                # Pearson相似度
                elif self.edge_weight_id == 2:
                    pearson = np.corrcoef(sentence_encoding[i].cpu().detach().numpy(), sentence_encoding[j].cpu().detach().numpy())[0][1]
                    weight = torch.tensor(pearson, dtype=torch.float).to(device)
                    edges_weight.append(weight)
                # 欧氏距离，1/分母作为权重
                elif self.edge_weight_id == 3:
                    distance = torch.pairwise_distance(sentence_encoding[i][None, :], sentence_encoding[j][None, :])
                    weight = 1 / (1 + distance[0])
                    edges_weight.append(weight)
                # kendall系数
                elif self.edge_weight_id == 4:
                    kendall = pd.Series(sentence_encoding[i].cpu().detach().numpy()).corr(
                        pd.Series(sentence_encoding[j].cpu().detach().numpy()), method="kendall")
                    weight = torch.tensor(kendall, dtype=torch.float).to(device)
                    edges_weight.append(weight)
                else:
                    print("wrong egde weight id")

                # whether to add another self-loop，这条边有ferature(1.)
                if self.gcn_loop:
                    if i == j:
                        edges.append((i, j))
                        edges_weight.append(weight)

        # 必须要先张量化
        edges = torch.tensor(edges)
        # 若超出实际长度则产生孤立节点
        graph = dgl.graph((edges[:, 0], edges[:, 1]), num_nodes=len(sentence_encoding)).to(device)

        # whether to add another self-loop，此时edges的feature为0
        # graph = dgl.add_self_loop(graph)

        edges_weight = torch.tensor(edges_weight).to(device)
        return graph, edges_weight


    # 测试集情况
    # document:(batch_n,doc_l,40,200)
    # pos:(batch_n,doc_l,6)  6个特征：['gpos', 'lpos', 'ppos', 'gid', 'lid', 'pid']
    # mask:NONE
    def forward(self, documents, pos=None, length_essay=None, device='cpu', mask=None):
        batch_n, doc_l, sen_l, _ = documents.size()  # documents: (batch_n, doc_l, sen_l, word_dim)
        self.init_hidden(batch_n=batch_n, doc_l=doc_l, device=device)
        documents = documents.view(batch_n * doc_l, sen_l, -1).transpose(0,1)  # documents: (sen_l, batch_n*doc_l, word_dim)

        sent_out, _ = self.sentLayer(documents, self.sent_hidden)  # sent_out: (sen_l, batch_n*doc_l, hidden_dim*2)
        sent_out = self.dropout(sent_out)

        if mask is None:
            # sentpres：(batch_n*doc_l,1,256)
            # 激活函数
            sentpres = torch.tanh(torch.mean(sent_out, dim=0))  # sentpres: (batch_n*doc_l, hidden_dim*2)
        else:
            sent_out = sent_out.masked_fill(mask.transpose(1, 0).unsqueeze(-1).expand_as(sent_out), 0)
            sentpres = torch.tanh(torch.sum(sent_out, dim=0) / (sen_l - mask.sum(dim=1).float() + 1e-9).unsqueeze(-1))

        sentpres = sentpres.view(batch_n, doc_l, self.hidden_dim * 2)  # sentpres: (batch_n, doc_l, hidden_dim*2)


        # 可以加dgl的位置Bottom
        # ---------------------------------------------------------------------

        # 加入GCN
        # length_essay
        # tensor([30, 30, 30, 30, 30, 30, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31,
        #         31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31,
        #         31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31],
        #        device='cuda:0')

        # 存储每篇文章经过dgl之后的feature
        total_sentence_feature = []

        for i in range(len(length_essay)):
            inner_sentennce = sentpres[i]
            node_length = length_essay[i]
            # build graph
            graph, edge_weight = self.build_graph(inner_sentennce, node_length, device=device)

            current_essay_sentence = [inner_sentennce]
            # try add different GCN，句间交互
            for sage_gcn in self.SAGE_GCN:
                # 输出：(node_nums, self.hidden_dim * 2)
                inner_sentennce = sage_gcn(graph, inner_sentennce, edge_weight)
                current_essay_sentence.append(inner_sentennce)
            # 每篇文章的
            current_essay_sentence = torch.cat(current_essay_sentence, dim=-1)  # current_essay_sentence:(node_nums, self.hidden_dim * 2 * dgl_layer)
            # 统一一下维度
            current_essay_sentence = self.transition_layer(current_essay_sentence)  # current_essay_sentence:(node_nums, self.hidden_dim * 2)
            # 加入总体中
            total_sentence_feature.append(current_essay_sentence)

        # 一个batch下的所有文章
        dgl_out = torch.stack(total_sentence_feature, dim=0)  # total_sentence_feature:(batch_size, node_nums, self.hidden_dim * 2)

        # ----------------------------------------------------------------------


        # sentence embedding的句间注意力
        sentFt = self.sfLayer(dgl_out)  # sentFt:(batch_n, doc_l,15)
        # sentFt = self.dropout(sentFt)


        # "add"情况下，将前三个pos位置1：1：1与sentence加和; ['gpos', 'lpos', 'ppos']
        sentpres = self.posLayer(dgl_out, pos)  # sentpres:(batch_n, doc_l, hidden_dim*2)

        sentpres = sentpres.transpose(0, 1)  # sentpres: (doc_l, batch_n, hidden_dim*2)

        tag_out, _ = self.tagLayer(sentpres, self.tag_hidden)  # tag_out: (doc_l, batch_n, sent_dim*2)
        tag_out = self.dropout(tag_out)

        tag_out = torch.tanh(tag_out)

        tag_out = tag_out.transpose(0, 1)  # tag_out: (batch_n, doc_l, sent_dim*2)


        # 可以加dgl的位置2(先加pos，再加dgl)



        roleFt = self.rfLayer(tag_out)  # roleFt:(batch_n, doc_l, 15)
        # roleFt = self.dropout(roleFt)

        new_out = torch.cat((tag_out, sentFt, roleFt), dim=2)  # tag_out: (batch_n, doc_l, sent_dim*2+30)  (1,8,286)
        # new_out = self.dropout(tag_out)

        # class_n用在了这里
        result = self.classifier(new_out)  # tag_out: (batch_n, doc_l, class_n)

        # log_softmax 和 nll_loss配套使用
        result = F.log_softmax(result, dim=2)  # result: (batch_n, doc_l, class_n)
        return result

    def getModelName(self):
        # pool_type='max_pool'的第一个字母m
        name = 'dgl_st_rs_spp%s' % self.pool_type[0]
        name += '_' + str(self.hidden_dim) + '_' + str(self.sent_dim)
        if self.p_embd == 'cat':
            name += '_cp'
        elif self.p_embd == 'add':
            name += '_ap'
        elif self.p_embd == 'embd':
            name += '_em'
        elif self.p_embd == 'embd_a':
            name += '_em_a'
        elif self.p_embd:
            name += '_' + self.p_embd
        return name