import torch
import torch.nn as nn
import torch.nn.functional as F

from subLayer import *


class STWithRSbySPP_GATE(nn.Module):
    def __init__(self, word_dim, hidden_dim, sent_dim, class_n, p_embd=None, pos_dim=0, p_embd_dim=16,
                 pool_type='max_pool'):
        # p_embd: 'cat', 'add','embd', 'embd_a'
        super(STWithRSbySPP_GATE, self).__init__()
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.sent_dim = sent_dim
        self.class_n = class_n
        self.p_embd = p_embd
        self.p_embd_dim = p_embd_dim
        self.pool_type = pool_type

        # gate的输入输出比例需要进行尝试，先尝试sent_dim*2+30到sent_dim
        self.gate1 = nn.Linear(self.sent_dim * 2 + 30, 3)

        self.gate2 = nn.Linear(15 * 2, 15)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=2)

        self.dropout = nn.Dropout(0.1)
        self.sentLayer = nn.LSTM(self.word_dim, self.hidden_dim, bidirectional=True)

        # gate2的情况
        self.classifier = nn.Linear(self.sent_dim * 2 + 15, self.class_n)
        # self.classifier = nn.Linear(self.sent_dim * 2 + 30, self.class_n)
        # self.classifier = nn.Linear(self.sent_dim * 2 + 60, self.class_n)

        self.posLayer = PositionLayer(p_embd, p_embd_dim)

        self.sfLayer = InterSentenceSPPLayer(self.hidden_dim * 2, pool_type=self.pool_type)
        self.rfLayer = InterSentenceSPPLayer(self.hidden_dim * 2, pool_type=self.pool_type)

        # self.sfLayer = InterSentenceSPPLayer3(self.hidden_dim*2, pool_type = self.pool_type)
        # self.rfLayer = InterSentenceSPPLayer3(self.hidden_dim*2, pool_type = self.pool_type)

        if p_embd == 'embd':
            self.tagLayer = nn.LSTM(self.hidden_dim * 2 + p_embd_dim * 3, self.sent_dim, bidirectional=True)
        elif p_embd == 'cat':
            self.tagLayer = nn.LSTM(self.hidden_dim * 2 + 3, self.sent_dim, bidirectional=True)
        else:
            self.tagLayer = nn.LSTM(self.hidden_dim * 2, self.sent_dim, bidirectional=True)

    # batch_size，一篇文章的句子个数
    def init_hidden(self, batch_n, doc_l, device='cpu'):
        self.sent_hidden = (torch.rand(2, batch_n * doc_l, self.hidden_dim, device=device).uniform_(-0.01, 0.01),
                            torch.rand(2, batch_n * doc_l, self.hidden_dim, device=device).uniform_(-0.01, 0.01))
        self.tag_hidden = (torch.rand(2, batch_n, self.sent_dim, device=device).uniform_(-0.01, 0.01),
                           torch.rand(2, batch_n, self.sent_dim, device=device).uniform_(-0.01, 0.01))

    # 测试集情况
    # document:(batch_n,doc_l,40,200)
    # pos:(batch_n,doc_l,6)  6个特征：['gpos', 'lpos', 'ppos', 'gid', 'lid', 'pid']
    # mask:NONE
    def forward(self, documents, pos=None, device='cpu', mask=None):
        batch_n, doc_l, sen_l, _ = documents.size()  # documents: (batch_n, doc_l, sen_l, word_dim)
        self.init_hidden(batch_n=batch_n, doc_l=doc_l, device=device)
        documents = documents.view(batch_n * doc_l, sen_l, -1).transpose(0,1)  # documents: (sen_l, batch_n*doc_l, word_dim)

        sent_out, _ = self.sentLayer(documents, self.sent_hidden)  # sent_out: (sen_l, batch_n*doc_l, hidden_dim*2)
        sent_out = self.dropout(sent_out)

        if mask is None:
            # sentpres：(batch_n*doc_l,1,256)
            sentpres = torch.tanh(torch.mean(sent_out, dim=0))  # sentpres: (batch_n*doc_l, hidden_dim*2)
        else:
            sent_out = sent_out.masked_fill(mask.transpose(1, 0).unsqueeze(-1).expand_as(sent_out), 0)
            sentpres = torch.tanh(torch.sum(sent_out, dim=0) / (sen_l - mask.sum(dim=1).float() + 1e-9).unsqueeze(-1))

        sentpres = sentpres.view(batch_n, doc_l, self.hidden_dim * 2)  # sentpres: (batch_n, doc_l, hidden_dim*2)

        # sentence embedding的句间注意力
        sentFt = self.sfLayer(sentpres)  # sentFt:(batch_n, doc_l,15)
        # sentFt = self.dropout(sentFt)

        sentpres = self.posLayer(sentpres, pos)  # sentpres:(batch_n, doc_l, hidden_dim*2)
        sentpres = sentpres.transpose(0, 1)  # sentpres: (doc_l, batch_n, hidden_dim*2)

        tag_out, _ = self.tagLayer(sentpres, self.tag_hidden)  # tag_out: (doc_l, batch_n, sent_dim*2)
        tag_out = self.dropout(tag_out)

        tag_out = torch.tanh(tag_out)

        tag_out = tag_out.transpose(0, 1)  # tag_out: (batch_n, doc_l, sent_dim*2)
        roleFt = self.rfLayer(tag_out)  # roleFt:(batch_n, doc_l, 15)
        # roleFt = self.dropout(roleFt)

        # 原先的
        # tag_out = torch.cat((tag_out, sentFt, roleFt), dim=2)  # tag_out: (batch_n, doc_l, sent_dim*2+30)  (1,8,286)

        # gate1
        # 相当于双线性层输出
        # tag_out_param = self.softmax(self.gate1(torch.cat((tag_out, sentFt, roleFt), dim=2)))
        # new_tag_out = torch.cat((tag_out_param[:, :, :1] * tag_out, tag_out_param[:, :, 1:2] * sentFt, tag_out_param[:, :, 2:3] * roleFt), dim=2)

        # gate2
        # 这种有点像加强了tag_out，因为sentFt和roleFt所占比重降低了
        gamma = self.sigmoid(self.gate2(torch.cat((sentFt, roleFt), dim=2)))
        new_value = gamma * sentFt + (1 - gamma) * roleFt
        new_tag_out = torch.cat((tag_out, new_value), dim=2)

        result = self.classifier(new_tag_out)  # tag_out: (batch_n, doc_l, class_n)

        result = F.log_softmax(result, dim=2)  # result: (batch_n, doc_l, class_n)
        return result

    def getModelName(self):
        # pool_type='max_pool'的第一个字母m
        name = 'st_rs_gru_spp%s' % self.pool_type[0]
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


class STWithRSbySPP_GRU_GATE(nn.Module):
    def __init__(self, word_dim, hidden_dim, sent_dim, class_n, p_embd=None, pos_dim=0, p_embd_dim=16,
                 pool_type='max_pool'):
        # p_embd: 'cat', 'add','embd', 'embd_a'
        super(STWithRSbySPP_GRU_GATE, self).__init__()
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.sent_dim = sent_dim
        self.class_n = class_n
        self.p_embd = p_embd
        self.p_embd_dim = p_embd_dim
        self.pool_type = pool_type

        # gate的输入输出比例需要进行尝试，先尝试sent_dim*2+30到sent_dim
        self.gate1 = nn.Linear(self.sent_dim * 2 + 30, 3)

        self.gate2 = nn.Linear(15 * 2, 15)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=2)

        self.dropout = nn.Dropout(0.1)
        self.sentLayer = nn.GRU(self.word_dim, self.hidden_dim, bidirectional=True)

        # gate2的情况
        self.classifier = nn.Linear(self.sent_dim * 2 + 15, self.class_n)
        # self.classifier = nn.Linear(self.sent_dim * 2 + 30, self.class_n)
        # self.classifier = nn.Linear(self.sent_dim * 2 + 60, self.class_n)

        self.posLayer = PositionLayer(p_embd, p_embd_dim)

        self.sfLayer = InterSentenceSPPLayer(self.hidden_dim * 2, pool_type=self.pool_type)
        self.rfLayer = InterSentenceSPPLayer(self.hidden_dim * 2, pool_type=self.pool_type)

        # self.sfLayer = InterSentenceSPPLayer3(self.hidden_dim*2, pool_type = self.pool_type)
        # self.rfLayer = InterSentenceSPPLayer3(self.hidden_dim*2, pool_type = self.pool_type)

        if p_embd == 'embd':
            self.tagLayer = nn.GRU(self.hidden_dim * 2 + p_embd_dim * 3, self.sent_dim, bidirectional=True)
        elif p_embd == 'cat':
            self.tagLayer = nn.GRU(self.hidden_dim * 2 + 3, self.sent_dim, bidirectional=True)
        else:
            self.tagLayer = nn.GRU(self.hidden_dim * 2, self.sent_dim, bidirectional=True)

    # batch_size，一篇文章的句子个数
    def init_hidden(self, batch_n, doc_l, device='cpu'):
        self.sent_hidden = torch.rand(2, batch_n * doc_l, self.hidden_dim, device=device).uniform_(-0.01, 0.01)
        self.tag_hidden = torch.rand(2, batch_n, self.sent_dim, device=device).uniform_(-0.01, 0.01)

    # 测试集情况
    # document:(batch_n,doc_l,40,200)
    # pos:(batch_n,doc_l,6)  6个特征：['gpos', 'lpos', 'ppos', 'gid', 'lid', 'pid']
    # mask:NONE
    def forward(self, documents, pos=None, device='cpu', mask=None):
        batch_n, doc_l, sen_l, _ = documents.size()  # documents: (batch_n, doc_l, sen_l, word_dim)
        self.init_hidden(batch_n=batch_n, doc_l=doc_l, device=device)
        documents = documents.view(batch_n * doc_l, sen_l, -1).transpose(0,1)  # documents: (sen_l, batch_n*doc_l, word_dim)

        sent_out, _ = self.sentLayer(documents, self.sent_hidden)  # sent_out: (sen_l, batch_n*doc_l, hidden_dim*2)
        sent_out = self.dropout(sent_out)

        if mask is None:
            # sentpres：(batch_n*doc_l,1,256)
            sentpres = torch.tanh(torch.mean(sent_out, dim=0))  # sentpres: (batch_n*doc_l, hidden_dim*2)
        else:
            sent_out = sent_out.masked_fill(mask.transpose(1, 0).unsqueeze(-1).expand_as(sent_out), 0)
            sentpres = torch.tanh(torch.sum(sent_out, dim=0) / (sen_l - mask.sum(dim=1).float() + 1e-9).unsqueeze(-1))

        sentpres = sentpres.view(batch_n, doc_l, self.hidden_dim * 2)  # sentpres: (batch_n, doc_l, hidden_dim*2)

        # sentence embedding的句间注意力
        sentFt = self.sfLayer(sentpres)  # sentFt:(batch_n, doc_l,15)
        # sentFt = self.dropout(sentFt)

        sentpres = self.posLayer(sentpres, pos)  # sentpres:(batch_n, doc_l, hidden_dim*2)
        sentpres = sentpres.transpose(0, 1)  # sentpres: (doc_l, batch_n, hidden_dim*2)

        tag_out, _ = self.tagLayer(sentpres, self.tag_hidden)  # tag_out: (doc_l, batch_n, sent_dim*2)
        tag_out = self.dropout(tag_out)

        tag_out = torch.tanh(tag_out)

        tag_out = tag_out.transpose(0, 1)  # tag_out: (batch_n, doc_l, sent_dim*2)
        roleFt = self.rfLayer(tag_out)  # roleFt:(batch_n, doc_l, 15)
        # roleFt = self.dropout(roleFt)

        # 原先的
        # tag_out = torch.cat((tag_out, sentFt, roleFt), dim=2)  # tag_out: (batch_n, doc_l, sent_dim*2+30)  (1,8,286)

        # gate1
        # 相当于双线性层输出
        # tag_out_param = self.softmax(self.gate1(torch.cat((tag_out, sentFt, roleFt), dim=2)))
        # new_tag_out = torch.cat((tag_out_param[:, :, :1] * tag_out, tag_out_param[:, :, 1:2] * sentFt, tag_out_param[:, :, 2:3] * roleFt), dim=2)

        # gate2
        # 这种有点像加强了tag_out，因为sentFt和roleFt所占比重降低了
        gamma = self.sigmoid(self.gate2(torch.cat((sentFt, roleFt), dim=2)))
        new_value = gamma * sentFt + (1 - gamma) * roleFt
        new_tag_out = torch.cat((tag_out, new_value), dim=2)

        result = self.classifier(new_tag_out)  # tag_out: (batch_n, doc_l, class_n)

        result = F.log_softmax(result, dim=2)  # result: (batch_n, doc_l, class_n)
        return result

    def getModelName(self):
        # pool_type='max_pool'的第一个字母m
        name = 'st_rs_gru_spp%s' % self.pool_type[0]
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



# 添加句子之间的attention，获取文章编码essay_encoding
class STWithRSbySPP_NewStructure1(nn.Module):
    def __init__(self, word_dim, hidden_dim, sent_dim, class_n, p_embd=None, pos_dim=0, p_embd_dim=16,
                 pool_type='max_pool'):
        # p_embd: 'cat', 'add','embd', 'embd_a'
        super(STWithRSbySPP_NewStructure1, self).__init__()
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.sent_dim = sent_dim
        self.class_n = class_n
        self.p_embd = p_embd
        self.p_embd_dim = p_embd_dim
        self.pool_type = pool_type


        self.softmax1 = nn.Softmax(dim=1)

        self.dropout = nn.Dropout(0.1)
        self.sentLayer = nn.LSTM(self.word_dim, self.hidden_dim, bidirectional=True)

        # new_structure1的情况
        # self.classifier = nn.Linear(self.sent_dim * 2, self.class_n)
        # gate2的情况
        # self.classifier = nn.Linear(self.sent_dim * 2 + 15, self.class_n)
        # baseline；new_structure1+2个SPP
        self.classifier = nn.Linear(self.sent_dim * 2 + 30, self.class_n)
        # self.classifier = nn.Linear(self.sent_dim * 2 + 60, self.class_n)

        self.posLayer = PositionLayer(p_embd, p_embd_dim)

        self.sfLayer = InterSentenceSPPLayer(self.hidden_dim * 2, pool_type=self.pool_type)
        self.rfLayer = InterSentenceSPPLayer(self.hidden_dim * 2, pool_type=self.pool_type)

        # self.sfLayer = InterSentenceSPPLayer3(self.hidden_dim*2, pool_type = self.pool_type)
        # self.rfLayer = InterSentenceSPPLayer3(self.hidden_dim*2, pool_type = self.pool_type)

        self.ws3 = nn.Linear(self.sent_dim * 2, self.sent_dim * 2)
        self.ws4 = nn.Linear(self.sent_dim * 2, 1)
        self.pre_pred = nn.Linear((self.sent_dim * 2 * 3), self.sent_dim * 2)

        if p_embd == 'embd':
            self.tagLayer = nn.LSTM(self.hidden_dim * 2 + p_embd_dim * 3, self.sent_dim, bidirectional=True)
        elif p_embd == 'cat':
            self.tagLayer = nn.LSTM(self.hidden_dim * 2 + 3, self.sent_dim, bidirectional=True)
        else:
            self.tagLayer = nn.LSTM(self.hidden_dim * 2, self.sent_dim, bidirectional=True)

    # batch_size，一篇文章的句子个数
    def init_hidden(self, batch_n, doc_l, device='cpu'):
        self.sent_hidden = (torch.rand(2, batch_n * doc_l, self.hidden_dim, device=device).uniform_(-0.01, 0.01),
                            torch.rand(2, batch_n * doc_l, self.hidden_dim, device=device).uniform_(-0.01, 0.01))
        self.tag_hidden = (torch.rand(2, batch_n, self.sent_dim, device=device).uniform_(-0.01, 0.01),
                           torch.rand(2, batch_n, self.sent_dim, device=device).uniform_(-0.01, 0.01))


    # 测试集情况
    # document:(batch_n,doc_l,40,200)
    # pos:(batch_n,doc_l,6)  6个特征：['gpos', 'lpos', 'ppos', 'gid', 'lid', 'pid']
    # mask:NONE
    def forward(self, documents, pos=None, device='cpu', mask=None):
        batch_n, doc_l, sen_l, _ = documents.size()  # documents: (batch_n, doc_l, sen_l, word_dim)
        self.init_hidden(batch_n=batch_n, doc_l=doc_l, device=device)
        documents = documents.view(batch_n * doc_l, sen_l, -1).transpose(0,1)  # documents: (sen_l, batch_n*doc_l, word_dim)

        sent_out, _ = self.sentLayer(documents, self.sent_hidden)  # sent_out: (sen_l, batch_n*doc_l, hidden_dim*2)
        sent_out = self.dropout(sent_out)

        if mask is None:
            # sentpres：(batch_n*doc_l,1,256)
            sentpres = torch.tanh(torch.mean(sent_out, dim=0))  # sentpres: (batch_n*doc_l, hidden_dim*2)
        else:
            sent_out = sent_out.masked_fill(mask.transpose(1, 0).unsqueeze(-1).expand_as(sent_out), 0)
            sentpres = torch.tanh(torch.sum(sent_out, dim=0) / (sen_l - mask.sum(dim=1).float() + 1e-9).unsqueeze(-1))

        sentpres = sentpres.view(batch_n, doc_l, self.hidden_dim * 2)  # sentpres: (batch_n, doc_l, hidden_dim*2)

        # sentence embedding的句间注意力
        sentFt = self.sfLayer(sentpres)  # sentFt:(batch_n, doc_l,15)
        # sentFt = self.dropout(sentFt)

        sentpres = self.posLayer(sentpres, pos)  # sentpres:(batch_n, doc_l, hidden_dim*2)
        sentpres = sentpres.transpose(0, 1)  # sentpres: (doc_l, batch_n, hidden_dim*2)

        # 增加一个维度
        tag_out, _ = self.tagLayer(sentpres, self.tag_hidden)  # tag_out: (doc_l, batch_n, sent_dim*2)
        # 记录一下句子的编码
        sent_encoding = tag_out.transpose(0,1)  # sent_encoding:(batch_n, doc_l, sent_dim*2)

        tag_out = self.dropout(tag_out)  # tag_out: (doc_l, batch_n, sent_dim*2)

        # ACL2020中，此处需要加一层全连接
        self_attention = self.ws3(tag_out)

        self_attention = torch.tanh(self_attention)
        self_attention = self_attention.transpose(0, 1)  # self_attention: (batch_n, doc_l, sent_dim*2)

        tag_out = torch.tanh(tag_out)
        tag_out = tag_out.transpose(0, 1)  # tag_out: (batch_n, doc_l, sent_dim*2)

        roleFt = self.rfLayer(tag_out)  # roleFt:(batch_n, doc_l, 15)
        # roleFt = self.dropout(roleFt)

        # squeeze()删除维度为1的
        self_attention = self.ws4(self.dropout(self_attention)).squeeze(2)  # self_attention: (batch_n, doc_l)
        self_attention = self.softmax1(self_attention)

        # *为逐乘；需要先增加维度，才能保证attention的数值是横着乘积的
        essay_encoding = torch.sum(sent_encoding*self_attention.unsqueeze(-1), dim=1)  # essay_encoding:(batch_n, sent_dim*2)

        essay_encoding = essay_encoding[:, None, :]   # essay_encoding:(batch_n, 1, sent_dim*2)

        # 扩展维度
        essay_encoding = essay_encoding.expand(sent_encoding.size())  #  essay_encoding:(batch_n, doc_l, sent_dim*2)

        out_s = torch.cat([sent_encoding, essay_encoding * sent_encoding, essay_encoding - sent_encoding], 2)

        pre_pred = F.tanh(self.pre_pred(self.dropout(out_s)))   # pre_pred:(batch_n, doc_l, sent_dim*2)

        pre_pred = self.dropout(pre_pred)

        # 原先的
        new_tagput = torch.cat((pre_pred, sentFt, roleFt), dim=2)  # tag_out: (batch_n, doc_l, sent_dim*2+30)  (1,8,286)

        # gate1
        # 相当于双线性层输出
        # tag_out_param = self.softmax(self.gate1(torch.cat((tag_out, sentFt, roleFt), dim=2)))
        # new_tag_out = torch.cat((tag_out_param[:, :, :1] * tag_out, tag_out_param[:, :, 1:2] * sentFt, tag_out_param[:, :, 2:3] * roleFt), dim=2)

        # gate2
        # 这种有点像加强了tag_out，因为sentFt和roleFt所占比重降低了
        # gamma = self.sigmoid(self.gate2(torch.cat((sentFt, roleFt), dim=2)))
        # new_value = gamma * sentFt + (1 - gamma) * roleFt
        # new_tag_out = torch.cat((tag_out, new_value), dim=2)

        result = self.classifier(new_tagput)  # tag_out: (batch_n, doc_l, class_n)

        result = F.log_softmax(result, dim=2)  # result: (batch_n, doc_l, class_n)
        return result

    def getModelName(self):
        # pool_type='max_pool'的第一个字母m
        name = 'new1_st_rs_gru_spp%s' % self.pool_type[0]
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




class STWithRSbySPP_NewStructure1_Gate2(nn.Module):
    def __init__(self, word_dim, hidden_dim, sent_dim, class_n, p_embd=None, pos_dim=0, p_embd_dim=16,
                 pool_type='max_pool'):
        # p_embd: 'cat', 'add','embd', 'embd_a'
        super(STWithRSbySPP_NewStructure1_Gate2, self).__init__()
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.sent_dim = sent_dim
        self.class_n = class_n
        self.p_embd = p_embd
        self.p_embd_dim = p_embd_dim
        self.pool_type = pool_type

        self.gate2 = nn.Linear(15 * 2, 15)

        self.sigmoid = nn.Sigmoid()
        self.softmax1 = nn.Softmax(dim=1)

        self.dropout = nn.Dropout(0.1)
        self.sentLayer = nn.LSTM(self.word_dim, self.hidden_dim, bidirectional=True)

        # new_structure1的情况
        # self.classifier = nn.Linear(self.sent_dim * 2, self.class_n)
        # gate2的情况
        self.classifier = nn.Linear(self.sent_dim * 2 + 15, self.class_n)
        # baseline；new_structure1+2个SPP
        # self.classifier = nn.Linear(self.sent_dim * 2 + 30, self.class_n)
        # self.classifier = nn.Linear(self.sent_dim * 2 + 60, self.class_n)

        self.posLayer = PositionLayer(p_embd, p_embd_dim)

        self.sfLayer = InterSentenceSPPLayer(self.hidden_dim * 2, pool_type=self.pool_type)
        self.rfLayer = InterSentenceSPPLayer(self.hidden_dim * 2, pool_type=self.pool_type)

        # self.sfLayer = InterSentenceSPPLayer3(self.hidden_dim*2, pool_type = self.pool_type)
        # self.rfLayer = InterSentenceSPPLayer3(self.hidden_dim*2, pool_type = self.pool_type)

        self.ws3 = nn.Linear(self.sent_dim * 2, self.sent_dim * 2)
        self.ws4 = nn.Linear(self.sent_dim * 2, 1)
        self.pre_pred = nn.Linear((self.sent_dim * 2 * 3), self.sent_dim * 2)

        if p_embd == 'embd':
            self.tagLayer = nn.LSTM(self.hidden_dim * 2 + p_embd_dim * 3, self.sent_dim, bidirectional=True)
        elif p_embd == 'cat':
            self.tagLayer = nn.LSTM(self.hidden_dim * 2 + 3, self.sent_dim, bidirectional=True)
        else:
            self.tagLayer = nn.LSTM(self.hidden_dim * 2, self.sent_dim, bidirectional=True)

    # batch_size，一篇文章的句子个数
    def init_hidden(self, batch_n, doc_l, device='cpu'):
        self.sent_hidden = (torch.rand(2, batch_n * doc_l, self.hidden_dim, device=device).uniform_(-0.01, 0.01),
                            torch.rand(2, batch_n * doc_l, self.hidden_dim, device=device).uniform_(-0.01, 0.01))
        self.tag_hidden = (torch.rand(2, batch_n, self.sent_dim, device=device).uniform_(-0.01, 0.01),
                           torch.rand(2, batch_n, self.sent_dim, device=device).uniform_(-0.01, 0.01))


    # 测试集情况
    # document:(batch_n,doc_l,40,200)
    # pos:(batch_n,doc_l,6)  6个特征：['gpos', 'lpos', 'ppos', 'gid', 'lid', 'pid']
    # mask:NONE
    def forward(self, documents, pos=None, device='cpu', mask=None):
        batch_n, doc_l, sen_l, _ = documents.size()  # documents: (batch_n, doc_l, sen_l, word_dim)
        self.init_hidden(batch_n=batch_n, doc_l=doc_l, device=device)
        documents = documents.view(batch_n * doc_l, sen_l, -1).transpose(0,1)  # documents: (sen_l, batch_n*doc_l, word_dim)

        sent_out, _ = self.sentLayer(documents, self.sent_hidden)  # sent_out: (sen_l, batch_n*doc_l, hidden_dim*2)
        sent_out = self.dropout(sent_out)

        if mask is None:
            # sentpres：(batch_n*doc_l,1,256)
            sentpres = torch.tanh(torch.mean(sent_out, dim=0))  # sentpres: (batch_n*doc_l, hidden_dim*2)
        else:
            sent_out = sent_out.masked_fill(mask.transpose(1, 0).unsqueeze(-1).expand_as(sent_out), 0)
            sentpres = torch.tanh(torch.sum(sent_out, dim=0) / (sen_l - mask.sum(dim=1).float() + 1e-9).unsqueeze(-1))

        sentpres = sentpres.view(batch_n, doc_l, self.hidden_dim * 2)  # sentpres: (batch_n, doc_l, hidden_dim*2)

        # sentence embedding的句间注意力
        sentFt = self.sfLayer(sentpres)  # sentFt:(batch_n, doc_l,15)
        # sentFt = self.dropout(sentFt)

        sentpres = self.posLayer(sentpres, pos)  # sentpres:(batch_n, doc_l, hidden_dim*2)
        sentpres = sentpres.transpose(0, 1)  # sentpres: (doc_l, batch_n, hidden_dim*2)

        tag_out, _ = self.tagLayer(sentpres, self.tag_hidden)  # tag_out: (doc_l, batch_n, sent_dim*2)
        tag_out = self.dropout(tag_out)  # tag_out: (doc_l, batch_n, sent_dim*2)

        # 记录一下句子的编码
        sent_encoding = tag_out.transpose(0, 1)  # sent_encoding:(batch_n, doc_l, sent_dim*2)

        # ACL2020中，此处需要加一层全连接
        self_attention = self.ws3(tag_out)

        self_attention = torch.tanh(self_attention)
        self_attention = self_attention.transpose(0, 1)  # self_attention: (batch_n, doc_l, sent_dim*2)

        tag_out = torch.tanh(tag_out)
        tag_out = tag_out.transpose(0, 1)  # tag_out: (batch_n, doc_l, sent_dim*2)

        roleFt = self.rfLayer(tag_out)  # roleFt:(batch_n, doc_l, 15)
        # roleFt = self.dropout(roleFt)

        # squeeze()删除维度为1的
        self_attention = self.ws4(self.dropout(self_attention)).squeeze(2)  # self_attention: (batch_n, doc_l)
        self_attention = self.softmax1(self_attention)

        # *为逐乘；需要先增加维度，才能保证attention的数值是横着乘积的
        essay_encoding = torch.sum(sent_encoding*self_attention.unsqueeze(-1), dim=1)  # essay_encoding:(batch_n, sent_dim*2)

        essay_encoding = essay_encoding[:, None, :]   # essay_encoding:(batch_n, 1, sent_dim*2)

        # 扩展维度
        essay_encoding = essay_encoding.expand(sent_encoding.size())  #  essay_encoding:(batch_n, doc_l, sent_dim*2)

        out_s = torch.cat([sent_encoding, essay_encoding * sent_encoding, essay_encoding - sent_encoding], 2)

        pre_pred = F.tanh(self.pre_pred(self.dropout(out_s)))   # pre_pred:(batch_n, doc_l, sent_dim*2)

        # pre_pred = self.dropout(pre_pred)

        # 先采用上面三个拼接，在两个SPP层拼接，之后两个结果拼接的方法
        gamma = self.sigmoid(self.gate2(torch.cat((sentFt, roleFt), dim=2)))
        new_value = gamma * sentFt + (1 - gamma) * roleFt  # new_value:(batch,doc_l,15)
        # new_tag_out = torch.cat((tag_out, new_value), dim=2)  # new_tag_out:(batch,doc_l,sent_dim*2+15)

        # 原先的
        new_output = torch.cat((pre_pred, new_value), dim=2)  # new_output: (batch_n, doc_l, sent_dim*2+15)

        # gate1
        # 相当于双线性层输出
        # tag_out_param = self.softmax(self.gate1(torch.cat((tag_out, sentFt, roleFt), dim=2)))
        # new_tag_out = torch.cat((tag_out_param[:, :, :1] * tag_out, tag_out_param[:, :, 1:2] * sentFt, tag_out_param[:, :, 2:3] * roleFt), dim=2)


        result = self.classifier(new_output)  # tag_out: (batch_n, doc_l, class_n)

        result = F.log_softmax(result, dim=2)  # result: (batch_n, doc_l, class_n)
        return result

    def getModelName(self):
        # pool_type='max_pool'的第一个字母m
        name = 'new1_st_rs_gru_spp%s' % self.pool_type[0]
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



class STWithRSbySPP_NewStructure1_Gate2_Cat2(nn.Module):
    def __init__(self, word_dim, hidden_dim, sent_dim, class_n, p_embd=None, pos_dim=0, p_embd_dim=16,
                 pool_type='max_pool'):
        # p_embd: 'cat', 'add','embd', 'embd_a'
        super(STWithRSbySPP_NewStructure1_Gate2_Cat2, self).__init__()
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.sent_dim = sent_dim
        self.class_n = class_n
        self.p_embd = p_embd
        self.p_embd_dim = p_embd_dim
        self.pool_type = pool_type

        self.gate2 = nn.Linear(15 * 2, 15)

        self.sigmoid = nn.Sigmoid()
        self.softmax1 = nn.Softmax(dim=1)

        self.dropout = nn.Dropout(0.1)
        self.sentLayer = nn.LSTM(self.word_dim, self.hidden_dim, bidirectional=True)

        # new_structure1的情况
        # self.classifier = nn.Linear(self.sent_dim * 2, self.class_n)
        # gate2的情况
        self.classifier = nn.Linear(self.sent_dim * 4 + 15, self.class_n)
        # baseline；new_structure1+2个SPP
        # self.classifier = nn.Linear(self.sent_dim * 2 + 30, self.class_n)
        # self.classifier = nn.Linear(self.sent_dim * 2 + 60, self.class_n)

        self.posLayer = PositionLayer(p_embd, p_embd_dim)

        self.sfLayer = InterSentenceSPPLayer(self.hidden_dim * 2, pool_type=self.pool_type)
        self.rfLayer = InterSentenceSPPLayer(self.hidden_dim * 2, pool_type=self.pool_type)

        # self.sfLayer = InterSentenceSPPLayer3(self.hidden_dim*2, pool_type = self.pool_type)
        # self.rfLayer = InterSentenceSPPLayer3(self.hidden_dim*2, pool_type = self.pool_type)

        self.ws3 = nn.Linear(self.sent_dim * 2, self.sent_dim * 2)
        self.ws4 = nn.Linear(self.sent_dim * 2, 1)
        self.pre_pred = nn.Linear((self.sent_dim * 2 * 3), self.sent_dim * 2)

        if p_embd == 'embd':
            self.tagLayer = nn.LSTM(self.hidden_dim * 2 + p_embd_dim * 3, self.sent_dim, bidirectional=True)
        elif p_embd == 'cat':
            self.tagLayer = nn.LSTM(self.hidden_dim * 2 + 3, self.sent_dim, bidirectional=True)
        else:
            self.tagLayer = nn.LSTM(self.hidden_dim * 2, self.sent_dim, bidirectional=True)

    # batch_size，一篇文章的句子个数
    def init_hidden(self, batch_n, doc_l, device='cpu'):
        self.sent_hidden = (torch.rand(2, batch_n * doc_l, self.hidden_dim, device=device).uniform_(-0.01, 0.01),
                            torch.rand(2, batch_n * doc_l, self.hidden_dim, device=device).uniform_(-0.01, 0.01))
        self.tag_hidden = (torch.rand(2, batch_n, self.sent_dim, device=device).uniform_(-0.01, 0.01),
                           torch.rand(2, batch_n, self.sent_dim, device=device).uniform_(-0.01, 0.01))


    # 测试集情况
    # document:(batch_n,doc_l,40,200)
    # pos:(batch_n,doc_l,6)  6个特征：['gpos', 'lpos', 'ppos', 'gid', 'lid', 'pid']
    # mask:NONE
    def forward(self, documents, pos=None, device='cpu', mask=None):
        batch_n, doc_l, sen_l, _ = documents.size()  # documents: (batch_n, doc_l, sen_l, word_dim)
        self.init_hidden(batch_n=batch_n, doc_l=doc_l, device=device)
        documents = documents.view(batch_n * doc_l, sen_l, -1).transpose(0,1)  # documents: (sen_l, batch_n*doc_l, word_dim)

        sent_out, _ = self.sentLayer(documents, self.sent_hidden)  # sent_out: (sen_l, batch_n*doc_l, hidden_dim*2)
        sent_out = self.dropout(sent_out)

        if mask is None:
            # sentpres：(batch_n*doc_l,1,256)
            sentpres = torch.tanh(torch.mean(sent_out, dim=0))  # sentpres: (batch_n*doc_l, hidden_dim*2)
        else:
            sent_out = sent_out.masked_fill(mask.transpose(1, 0).unsqueeze(-1).expand_as(sent_out), 0)
            sentpres = torch.tanh(torch.sum(sent_out, dim=0) / (sen_l - mask.sum(dim=1).float() + 1e-9).unsqueeze(-1))

        sentpres = sentpres.view(batch_n, doc_l, self.hidden_dim * 2)  # sentpres: (batch_n, doc_l, hidden_dim*2)

        # sentence embedding的句间注意力
        sentFt = self.sfLayer(sentpres)  # sentFt:(batch_n, doc_l,15)
        # sentFt = self.dropout(sentFt)

        sentpres = self.posLayer(sentpres, pos)  # sentpres:(batch_n, doc_l, hidden_dim*2)
        sentpres = sentpres.transpose(0, 1)  # sentpres: (doc_l, batch_n, hidden_dim*2)

        # 增加一个维度
        tag_out, _ = self.tagLayer(sentpres, self.tag_hidden)  # tag_out: (doc_l, batch_n, sent_dim*2)
        tag_out = self.dropout(tag_out)  # tag_out: (doc_l, batch_n, sent_dim*2)

        # 记录一下句子的编码
        sent_encoding = tag_out.transpose(0, 1)  # sent_encoding:(batch_n, doc_l, sent_dim*2)

        # ACL2020中，此处需要加一层全连接
        self_attention = self.ws3(tag_out)

        self_attention = torch.tanh(self_attention)
        self_attention = self_attention.transpose(0, 1)  # self_attention: (batch_n, doc_l, sent_dim*2)

        tag_out = torch.tanh(tag_out)
        tag_out = tag_out.transpose(0, 1)  # tag_out: (batch_n, doc_l, sent_dim*2)

        roleFt = self.rfLayer(tag_out)  # roleFt:(batch_n, doc_l, 15)
        # roleFt = self.dropout(roleFt)

        # squeeze()删除维度为1的
        self_attention = self.ws4(self.dropout(self_attention)).squeeze(2)  # self_attention: (batch_n, doc_l)
        self_attention = self.softmax1(self_attention)

        # *为逐乘；需要先增加维度，才能保证attention的数值是横着乘积的
        essay_encoding = torch.sum(sent_encoding*self_attention.unsqueeze(-1), dim=1)  # essay_encoding:(batch_n, sent_dim*2)

        essay_encoding = essay_encoding[:, None, :]   # essay_encoding:(batch_n, 1, sent_dim*2)

        # 扩展维度
        essay_encoding = essay_encoding.expand(sent_encoding.size())  #  essay_encoding:(batch_n, doc_l, sent_dim*2)

        out_s = torch.cat([sent_encoding, essay_encoding * sent_encoding, essay_encoding - sent_encoding], 2)

        pre_pred = F.tanh(self.pre_pred(self.dropout(out_s)))   # pre_pred:(batch_n, doc_l, sent_dim*2)

        # pre_pred = self.dropout(pre_pred)

        # 先采用上面三个拼接，在两个SPP层拼接，之后两个结果拼接的方法
        gamma = self.sigmoid(self.gate2(torch.cat((sentFt, roleFt), dim=2)))
        new_value = gamma * sentFt + (1 - gamma) * roleFt  # new_value:(batch,doc_l,15)
        new_tag_out = torch.cat((tag_out, new_value), dim=2)  # new_tag_out:(batch,doc_l,sent_dim*2+15)

        # 原先的
        new_output = torch.cat((pre_pred, new_tag_out), dim=2)  # new_output: (batch_n, doc_l, sent_dim*4+15)

        # gate1
        # 相当于双线性层输出
        # tag_out_param = self.softmax(self.gate1(torch.cat((tag_out, sentFt, roleFt), dim=2)))
        # new_tag_out = torch.cat((tag_out_param[:, :, :1] * tag_out, tag_out_param[:, :, 1:2] * sentFt, tag_out_param[:, :, 2:3] * roleFt), dim=2)


        result = self.classifier(new_output)  # tag_out: (batch_n, doc_l, class_n)

        result = F.log_softmax(result, dim=2)  # result: (batch_n, doc_l, class_n)
        return result

    def getModelName(self):
        # pool_type='max_pool'的第一个字母m
        name = 'new1_st_rs_gru_spp%s' % self.pool_type[0]
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



# new_baseline
# 英文的门控机制用LSTM+dropout0.1(SPP2)才测试
# class EnSTWithRSbySPP_GATE(nn.Module):
#     def __init__(self, word_dim, hidden_dim, sent_dim, class_n, p_embd=None, pos_dim=0, p_embd_dim=16,
#                  pool_type='max_pool'):
#         # p_embd: 'cat', 'add','embd', 'embd_a'
#         super(EnSTWithRSbySPP_GATE, self).__init__()
#         self.word_dim = word_dim
#         self.hidden_dim = hidden_dim
#         self.sent_dim = sent_dim
#         self.class_n = class_n
#         self.p_embd = p_embd
#         self.p_embd_dim = p_embd_dim
#         self.pool_type = pool_type
#
#         # gate的输入输出比例需要进行尝试，先尝试sent_dim*2+30到sent_dim
#         self.gate1 = nn.Linear(self.sent_dim * 2 + 30, 3)
#
#         self.gate2 = nn.Linear(15 * 2, 15)
#
#         self.sigmoid = nn.Sigmoid()
#         self.softmax = nn.Softmax(dim=2)
#
#         self.dropout = nn.Dropout(0.1)
#
#         self.sentLayer = nn.LSTM(self.word_dim, self.hidden_dim, bidirectional=True)
#         # self.classifier = nn.Linear(self.sent_dim * 2 + 30, self.class_n)
#         self.classifier = nn.Linear(self.sent_dim * 2 + 15, self.class_n)
#
#         self.posLayer = PositionLayer(p_embd, p_embd_dim)
#
#         self.sfLayer = InterSentenceSPPLayer(self.hidden_dim * 2, pool_type=self.pool_type)
#         self.rfLayer = InterSentenceSPPLayer(self.hidden_dim * 2, pool_type=self.pool_type)
#
#         if p_embd == 'embd':
#             self.tagLayer = nn.LSTM(self.hidden_dim * 2 + p_embd_dim * 3, self.sent_dim, bidirectional=True)
#         elif p_embd == 'cat':
#             self.tagLayer = nn.LSTM(self.hidden_dim * 2 + 3, self.sent_dim, bidirectional=True)
#         else:
#             self.tagLayer = nn.LSTM(self.hidden_dim * 2, self.sent_dim, bidirectional=True)
#
#     # batch_size，一篇文章的句子个数
#     def init_hidden(self, batch_n, doc_l, device='cpu'):
#
#         self.sent_hidden = (torch.rand(2, batch_n * doc_l, self.hidden_dim, device=device).uniform_(-0.01, 0.01),
#                             torch.rand(2, batch_n * doc_l, self.hidden_dim, device=device).uniform_(-0.01, 0.01))
#         # 两个元组，shape都是(2,batch_n,hidden_dim)
#         self.tag_hidden = (torch.rand(2, batch_n, self.sent_dim, device=device).uniform_(-0.01, 0.01),
#                            torch.rand(2, batch_n, self.sent_dim, device=device).uniform_(-0.01, 0.01))
#
#     # 测试集情况
#     # document:(batch_n,doc_l,40,200)
#     # pos:(batch_n,doc_l,6)  6个特征：['gpos', 'lpos', 'ppos', 'gid', 'lid', 'pid']
#     # mask:NONE
#     def forward(self, documents, pos=None, device='cpu', mask=None):
#         batch_n, doc_l, sen_l, _ = documents.size()  # documents: (batch_n, doc_l, sen_l, word_dim)
#         self.init_hidden(batch_n=batch_n, doc_l=doc_l, device=device)
#         documents = documents.view(batch_n * doc_l, sen_l, -1).transpose(0,1)  # documents: (sen_l, batch_n*doc_l, word_dim)
#
#         sent_out, _ = self.sentLayer(documents, self.sent_hidden)  # sent_out: (sen_l, batch_n*doc_l, hidden_dim*2)
#
#         if mask is None:
#             # sentpres：(batch_n*doc_l,1,256)
#             # 激活函数
#             sentpres = torch.tanh(torch.mean(sent_out, dim=0))  # sentpres: (batch_n*doc_l, hidden_dim*2)
#         else:
#             sent_out = sent_out.masked_fill(mask.transpose(1, 0).unsqueeze(-1).expand_as(sent_out), 0)
#             sentpres = torch.tanh(torch.sum(sent_out, dim=0) / (sen_l - mask.sum(dim=1).float() + 1e-9).unsqueeze(-1))
#
#         sentpres = sentpres.view(batch_n, doc_l, self.hidden_dim * 2)  # sentpres: (batch_n, doc_l, hidden_dim*2)
#
#         # sentence embedding的句间注意力
#         sentFt = self.sfLayer(sentpres)  # sentFt:(batch_n, doc_l,15)
#         sentFt = self.dropout(sentFt)
#
#         # "add"情况下，将前三个pos位置1：1：1与sentence加和; ['gpos', 'lpos', 'ppos']
#         sentpres = self.posLayer(sentpres, pos)  # sentpres:(batch_n, doc_l, hidden_dim*2)
#         sentpres = sentpres.transpose(0, 1)  # sentpres: (doc_l, batch_n, hidden_dim*2)
#
#         tag_out, _ = self.tagLayer(sentpres, self.tag_hidden)  # tag_out: (doc_l, batch_n, sent_dim*2)
#
#         tag_out = torch.tanh(tag_out)
#
#         tag_out = tag_out.transpose(0, 1)  # tag_out: (batch_n, doc_l, sent_dim*2)
#         roleFt = self.rfLayer(tag_out)  # roleFt:(batch_n, doc_l, 15)
#         roleFt = self.dropout(roleFt)
#
#         # 原先的
#         # tag_out = torch.cat((tag_out, sentFt, roleFt), dim=2)  # tag_out: (batch_n, doc_l, sent_dim*2+30)  (1,8,286)
#         # 相当于双线性层输出
#         # tag_out_param = self.softmax(self.gate1(torch.cat((tag_out, sentFt, roleFt), dim=2)))
#         # new_tag_out = torch.cat((tag_out_param[:, :, :1] * tag_out, tag_out_param[:, :, 1:2] * sentFt, tag_out_param[:, :, 2:3] * roleFt), dim=2)
#
#         gamma = self.sigmoid(self.gate2(torch.cat((sentFt, roleFt), dim=2)))
#         new_value = gamma * sentFt + (1 - gamma) * roleFt
#         new_tag_out = torch.cat((tag_out, new_value), dim=2)
#
#         # class_n用在了这里
#         result = self.classifier(new_tag_out)  # tag_out: (batch_n, doc_l, class_n)
#
#         # log_softmax 和 nll_loss配套使用
#         result = F.log_softmax(result, dim=2)  # result: (batch_n, doc_l, class_n)
#         return result
#
#     def getModelName(self):
#         # pool_type='max_pool'的第一个字母m
#         name = 'st_rs_spp%s' % self.pool_type[0]
#         name += '_' + str(self.hidden_dim) + '_' + str(self.sent_dim)
#         if self.p_embd == 'cat':
#             name += '_cp'
#         elif self.p_embd == 'add':
#             name += '_ap'
#         elif self.p_embd == 'embd':
#             name += '_em'
#         elif self.p_embd == 'embd_a':
#             name += '_em_a'
#         elif self.p_embd:
#             name += '_' + self.p_embd
#         return name



# 英文+feature用纯GRU进行测试
# class STWithRSbySPPWithFt2_GATE(nn.Module):
#     def __init__(self, word_dim, hidden_dim, sent_dim, class_n, p_embd=None, pos_dim=0, p_embd_dim=16, ft_size=0,
#                  pool_type='max_pool'):
#         # p_embd: 'cat', 'add','embd', 'embd_a'
#         super(STWithRSbySPPWithFt2_GATE, self).__init__()
#         self.word_dim = word_dim
#         self.hidden_dim = hidden_dim
#         self.sent_dim = sent_dim
#         self.class_n = class_n
#         self.p_embd = p_embd
#         self.p_embd_dim = p_embd_dim
#         self.ft_size = ft_size
#         self.pool_type = pool_type
#
#         self.gate1 = nn.Linear(self.sent_dim * 2 + 30, 3)
#
#         self.gate2 = nn.Linear(15 * 2, 15)
#
#         self.sigmoid = nn.Sigmoid()
#         self.softmax = nn.Softmax(dim=2)
#
#         self.sentLayer = nn.GRU(self.word_dim, self.hidden_dim, bidirectional=True)
#
#         self.classifier = nn.Linear(self.sent_dim * 2 + 15, self.class_n)
#
#         self.posLayer = PositionLayer(p_embd, p_embd_dim)
#         self.sfLayer = InterSentenceSPPLayer(self.hidden_dim * 2, pool_type=self.pool_type)
#         self.rfLayer = InterSentenceSPPLayer(self.hidden_dim * 2, pool_type=self.pool_type)
#
#         if p_embd == 'embd':
#             self.tagLayer = nn.GRU(self.hidden_dim * 2 + p_embd_dim * 3 + ft_size, self.sent_dim, bidirectional=True)
#         elif p_embd == 'cat':
#             self.tagLayer = nn.GRU(self.hidden_dim * 2 + 3 + ft_size, self.sent_dim, bidirectional=True)
#         else:
#             self.tagLayer = nn.GRU(self.hidden_dim * 2 + ft_size, self.sent_dim, bidirectional=True)
#
#     def init_hidden(self, batch_n, doc_l, device='cpu'):
#         # h0: [num_layers * num_directions, batch, hidden_size]
#         self.sent_hidden = torch.rand(2, batch_n * doc_l, self.hidden_dim, device=device).uniform_(-0.01, 0.01)
#         self.tag_hidden = torch.rand(2, batch_n, self.sent_dim, device=device).uniform_(-0.01, 0.01)
#
#     # inputs:(30,25,40,768)
#     # tp:(30,25,6)  前六个基础特征
#     # tft:(30,25,9)  后九个新增的特征
#     def forward(self, documents, pos, ft, device='cpu', mask=None):
#         # 保证数据的有效性
#         ft = ft[:, :, :self.ft_size]   # ft:(batch_n,doc_l,9)
#         batch_n, doc_l, sen_l, _ = documents.size()  # documents: (batch_n, doc_l, sen_l, word_dim)
#         self.init_hidden(batch_n=batch_n, doc_l=doc_l, device=device)
#         documents = documents.view(batch_n * doc_l, sen_l, -1).transpose(0,1)  # documents: (sen_l, batch_n*doc_l, word_dim)
#
#         sent_out, _ = self.sentLayer(documents, self.sent_hidden)  # sent_out: (sen_l, batch_n*doc_l, hidden_dim*2)
#
#         if mask is None:
#             sentpres = torch.tanh(torch.mean(sent_out, dim=0))  # sentpres: (batch_n*doc_l, hidden_dim*2)
#         else:
#             sent_out = sent_out.masked_fill(mask.transpose(1, 0).unsqueeze(-1).expand_as(sent_out), 0)
#             sentpres = torch.tanh(torch.sum(sent_out, dim=0) / (sen_l - mask.sum(dim=1).float() + 1e-9).unsqueeze(-1))
#
#         sentpres = sentpres.view(batch_n, doc_l, self.hidden_dim * 2)  # sentpres: (batch_n, doc_l, hidden_dim*2)
#         # sentence embedding的句间注意力
#         sentFt = self.sfLayer(sentpres)  # sentFt:(batch_n, doc_l,15)
#
#         # 添加SPE
#         sentpres = self.posLayer(sentpres, pos)  # sentpres:(batch_n, doc_l, hidden_dim*2)
#         # 添加特征feature进去
#         sentpres = torch.cat((sentpres, ft), dim=2)  # sentpres:(batch_n, doc_l, hidden_dim*2+ft_size)
#
#         sentpres = sentpres.transpose(0, 1)
#
#         tag_out, _ = self.tagLayer(sentpres, self.tag_hidden)  # tag_out: (doc_l, batch_n, sent_dim*2)
#
#         tag_out = torch.tanh(tag_out)
#
#         tag_out = tag_out.transpose(0, 1)
#         roleFt = self.rfLayer(tag_out)
#
#         # 旧的
#         # tag_out = torch.cat((tag_out, sentFt, roleFt), dim=2)
#
#         # 1、相当于双线性层输出
#         # tag_out_param = self.softmax(self.gate1(torch.cat((tag_out, sentFt, roleFt), dim=2)))
#         # new_tag_out = torch.cat((tag_out_param[:, :, :1] * tag_out, tag_out_param[:, :, 1:2] * sentFt, tag_out_param[:, :, 2:3] * roleFt), dim=2)
#
#         gamma = self.sigmoid(self.gate2(torch.cat((sentFt, roleFt), dim=2)))
#         new_value = gamma * sentFt + (1 - gamma) * roleFt
#         new_tag_out = torch.cat((tag_out, new_value), dim=2)
#
#         result = self.classifier(new_tag_out)
#
#         result = F.log_softmax(result, dim=2)  # result: (batch_n, doc_l, class_n)
#         return result
#
#     def getModelName(self):
#         name = 'sent_gru_%s_ft_%d' % (self.pool_type[0], self.ft_size)
#         name += '_' + str(self.hidden_dim) + '_' + str(self.sent_dim) + '_' + str(self.ft_size)
#         if self.p_embd == 'cat':
#             name += '_cp'
#         elif self.p_embd == 'add':
#             name += '_ap'
#         elif self.p_embd == 'embd':
#             name += '_em'
#         elif self.p_embd == 'embd_a':
#             name += '_em_a'
#         elif self.p_embd:
#             name += '_' + self.p_embd
#         return name