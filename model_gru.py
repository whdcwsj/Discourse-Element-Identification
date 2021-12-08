import torch
import torch.nn as nn
import torch.nn.functional as F

from subLayer import *


class STWithRSbySPP_GRU(nn.Module):
    def __init__(self, word_dim, hidden_dim, sent_dim, class_n, p_embd=None, pos_dim=0, p_embd_dim=16,
                 pool_type='max_pool'):
        # p_embd: 'cat', 'add','embd', 'embd_a'
        super(STWithRSbySPP_GRU, self).__init__()
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.sent_dim = sent_dim
        self.class_n = class_n
        self.p_embd = p_embd
        self.p_embd_dim = p_embd_dim
        self.pool_type = pool_type

        self.dropout = nn.Dropout(0.1)
        self.sentLayer = nn.GRU(self.word_dim, self.hidden_dim, bidirectional=True)

        self.classifier = nn.Linear(self.sent_dim * 2 + 30, self.class_n)
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
        # h0: [num_layers * num_directions, batch, hidden_size]
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
        # sent_out = self.dropout(sent_out)

        if mask is None:
            # sentpres：(batch_n*doc_l,1,256)
            sentpres = torch.tanh(torch.mean(sent_out, dim=0))  # sentpres: (batch_n*doc_l, hidden_dim*2)
        else:
            sent_out = sent_out.masked_fill(mask.transpose(1, 0).unsqueeze(-1).expand_as(sent_out), 0)
            sentpres = torch.tanh(torch.sum(sent_out, dim=0) / (sen_l - mask.sum(dim=1).float() + 1e-9).unsqueeze(-1))

        sentpres = sentpres.view(batch_n, doc_l, self.hidden_dim * 2)  # sentpres: (batch_n, doc_l, hidden_dim*2)

        # sentence embedding的句间注意力
        sentFt = self.sfLayer(sentpres)  # sentFt:(batch_n, doc_l,15)
        sentFt = self.dropout(sentFt)

        sentpres = self.posLayer(sentpres, pos)  # sentpres:(batch_n, doc_l, hidden_dim*2)
        sentpres = sentpres.transpose(0, 1)  # sentpres: (doc_l, batch_n, hidden_dim*2)

        tag_out, _ = self.tagLayer(sentpres, self.tag_hidden)  # tag_out: (doc_l, batch_n, sent_dim*2)
        # tag_out = self.dropout(tag_out)

        tag_out = torch.tanh(tag_out)

        tag_out = tag_out.transpose(0, 1)  # tag_out: (batch_n, doc_l, sent_dim*2)
        roleFt = self.rfLayer(tag_out)  # roleFt:(batch_n, doc_l, 15)
        roleFt = self.dropout(roleFt)

        tag_out = torch.cat((tag_out, sentFt, roleFt), dim=2)  # tag_out: (batch_n, doc_l, sent_dim*2+30)  (1,8,286)
        # tag_out = self.dropout(tag_out)

        result = self.classifier(tag_out)  # tag_out: (batch_n, doc_l, class_n)

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



# 将基本的LSTM模型替换为GRU
class EnSTWithRSbySPP_GRU(nn.Module):
    def __init__(self, word_dim, hidden_dim, sent_dim, class_n, p_embd=None, pos_dim=0, p_embd_dim=16,
                 pool_type='max_pool'):
        # p_embd: 'cat', 'add','embd', 'embd_a'
        super(EnSTWithRSbySPP_GRU, self).__init__()
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.sent_dim = sent_dim
        self.class_n = class_n
        self.p_embd = p_embd
        self.p_embd_dim = p_embd_dim
        self.pool_type = pool_type

        self.dropout = nn.Dropout(0.1)
        self.sentLayer = nn.GRU(self.word_dim, self.hidden_dim, bidirectional=True)

        self.classifier = nn.Linear(self.sent_dim * 2 + 30, self.class_n)

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
        # h0: [num_layers * num_directions, batch, hidden_size]
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
            sentpres = torch.tanh(torch.mean(sent_out, dim=0))  # sentpres: (batch_n*doc_l, hidden_dim*2)
        else:
            sent_out = sent_out.masked_fill(mask.transpose(1, 0).unsqueeze(-1).expand_as(sent_out), 0)
            sentpres = torch.tanh(torch.sum(sent_out, dim=0) / (sen_l - mask.sum(dim=1).float() + 1e-9).unsqueeze(-1))

        sentpres = sentpres.view(batch_n, doc_l, self.hidden_dim * 2)  # sentpres: (batch_n, doc_l, hidden_dim*2)

        # sentence embedding的句间注意力
        sentFt = self.sfLayer(sentpres)  # sentFt:(batch_n, doc_l,15)
        # sentFt = self.dropout(sentFt)

        sentpres = self.posLayer(sentpres, pos)   # sentpres:(batch_n, doc_l, hidden_dim*2)
        sentpres = sentpres.transpose(0, 1)   # sentpres: (doc_l, batch_n, hidden_dim*2)

        tag_out, _ = self.tagLayer(sentpres, self.tag_hidden)  # tag_out: (doc_l, batch_n, sent_dim*2)
        tag_out = self.dropout(tag_out)

        tag_out = torch.tanh(tag_out)

        tag_out = tag_out.transpose(0, 1)  # tag_out: (batch_n, doc_l, sent_dim*2)
        roleFt = self.rfLayer(tag_out)  # roleFt:(batch_n, doc_l, 15)
        # roleFt = self.dropout(roleFt)

        tag_out = torch.cat((tag_out, sentFt, roleFt), dim=2)  # tag_out: (batch_n, doc_l, sent_dim*2+30)  (1,8,286)
        # tag_out = self.dropout(tag_out)

        # class_n用在了这里
        result = self.classifier(tag_out)  # tag_out: (batch_n, doc_l, class_n)

        # log_softmax 和 nll_loss配套使用
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



class STWithRSbySPPWithFt2_GRU(nn.Module):
    def __init__(self, word_dim, hidden_dim, sent_dim, class_n, p_embd=None, pos_dim=0, p_embd_dim=16, ft_size=0,
                 pool_type='max_pool'):
        # p_embd: 'cat', 'add','embd', 'embd_a'
        super(STWithRSbySPPWithFt2_GRU, self).__init__()
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.sent_dim = sent_dim
        self.class_n = class_n
        self.p_embd = p_embd
        self.p_embd_dim = p_embd_dim
        self.ft_size = ft_size
        self.pool_type = pool_type

        self.dropout = nn.Dropout(0.1)
        self.sentLayer = nn.GRU(self.word_dim, self.hidden_dim, bidirectional=True)

        self.classifier = nn.Linear(self.sent_dim * 2 + 30, self.class_n)

        self.posLayer = PositionLayer(p_embd, p_embd_dim)
        self.sfLayer = InterSentenceSPPLayer(self.hidden_dim * 2, pool_type=self.pool_type)
        self.rfLayer = InterSentenceSPPLayer(self.hidden_dim * 2, pool_type=self.pool_type)

        if p_embd == 'embd':
            self.tagLayer = nn.GRU(self.hidden_dim * 2 + p_embd_dim * 3 + ft_size, self.sent_dim, bidirectional=True)
        elif p_embd == 'cat':
            self.tagLayer = nn.GRU(self.hidden_dim * 2 + 3 + ft_size, self.sent_dim, bidirectional=True)
        else:
            self.tagLayer = nn.GRU(self.hidden_dim * 2 + ft_size, self.sent_dim, bidirectional=True)

    def init_hidden(self, batch_n, doc_l, device='cpu'):
        # h0: [num_layers * num_directions, batch, hidden_size]
        self.sent_hidden = torch.rand(2, batch_n * doc_l, self.hidden_dim, device=device).uniform_(-0.01, 0.01)
        self.tag_hidden = torch.rand(2, batch_n, self.sent_dim, device=device).uniform_(-0.01, 0.01)

    # inputs:(30,25,40,768)
    # tp:(30,25,6)  前六个基础特征
    # tft:(30,25,9)  后九个新增的特征
    def forward(self, documents, pos, ft, device='cpu', mask=None):
        # 保证数据的有效性
        ft = ft[:, :, :self.ft_size]   # ft:(batch_n,doc_l,9)
        batch_n, doc_l, sen_l, _ = documents.size()  # documents: (batch_n, doc_l, sen_l, word_dim)
        self.init_hidden(batch_n=batch_n, doc_l=doc_l, device=device)
        documents = documents.view(batch_n * doc_l, sen_l, -1).transpose(0,
                                                                         1)  # documents: (sen_l, batch_n*doc_l, word_dim)

        sent_out, _ = self.sentLayer(documents, self.sent_hidden)  # sent_out: (sen_l, batch_n*doc_l, hidden_dim*2)
        # sent_out = self.dropout(sent_out)

        if mask is None:
            sentpres = torch.tanh(torch.mean(sent_out, dim=0))  # sentpres: (batch_n*doc_l, hidden_dim*2)
        else:
            sent_out = sent_out.masked_fill(mask.transpose(1, 0).unsqueeze(-1).expand_as(sent_out), 0)
            sentpres = torch.tanh(torch.sum(sent_out, dim=0) / (sen_l - mask.sum(dim=1).float() + 1e-9).unsqueeze(-1))

        sentpres = sentpres.view(batch_n, doc_l, self.hidden_dim * 2)  # sentpres: (batch_n, doc_l, hidden_dim*2)
        # sentence embedding的句间注意力
        sentFt = self.sfLayer(sentpres)  # sentFt:(batch_n, doc_l,15)
        sentFt = self.dropout(sentFt)

        # 添加SPE
        sentpres = self.posLayer(sentpres, pos)  # sentpres:(batch_n, doc_l, hidden_dim*2)
        # 添加特征feature进去
        sentpres = torch.cat((sentpres, ft), dim=2)  # sentpres:(batch_n, doc_l, hidden_dim*2+ft_size)

        sentpres = sentpres.transpose(0, 1)

        tag_out, _ = self.tagLayer(sentpres, self.tag_hidden)  # tag_out: (doc_l, batch_n, sent_dim*2)
        # tag_out = self.dropout(tag_out)

        tag_out = torch.tanh(tag_out)

        tag_out = tag_out.transpose(0, 1)
        roleFt = self.rfLayer(tag_out)
        roleFt = self.dropout(roleFt)

        tag_out = torch.cat((tag_out, sentFt, roleFt), dim=2)
        # tag_out = self.dropout(tag_out)

        result = self.classifier(tag_out)

        result = F.log_softmax(result, dim=2)  # result: (batch_n, doc_l, class_n)
        return result

    def getModelName(self):
        name = 'sent_gru_%s_ft_%d' % (self.pool_type[0], self.ft_size)
        name += '_' + str(self.hidden_dim) + '_' + str(self.sent_dim) + '_' + str(self.ft_size)
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