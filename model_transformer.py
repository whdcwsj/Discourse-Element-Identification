import torch
import torch.nn as nn
import torch.nn.functional as F

from subLayer import *


# type_id=0
# 用于中文数据集的情况下：
# 输入：vector_size=200，hidden_size=128，sent_dim=128，class_n=8，p_embd='add'，p_embd_dim=16，pool_type='max_pool'
# 英文数据集：
# p_embd = None
class STWithRSbySPP_Transformer(nn.Module):
    def __init__(self, word_dim, hidden_dim, sent_dim, class_n, p_embd=None, pos_dim=0, p_embd_dim=16,
                 pool_type='max_pool'):
        # p_embd: 'cat', 'add','embd', 'embd_a'
        super(STWithRSbySPP_Transformer, self).__init__()
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.sent_dim = sent_dim
        self.class_n = class_n
        self.p_embd = p_embd
        self.p_embd_dim = p_embd_dim
        self.pool_type = pool_type

        # self.dropout = nn.Dropout(0.1)
        self.sentLayer = nn.LSTM(self.word_dim, self.hidden_dim, bidirectional=True)
        # 为什么sent_dim*2+30？？？因为要接两个句间注意力
        self.classifier = nn.Linear(self.sent_dim * 2 + 30, self.class_n)

        # 配合avg与max加和时进行使用
        # self.classifier = nn.Linear(self.sent_dim * 2 + 60, self.class_n)

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

    # batch_size，一篇文章的句子个数
    def init_hidden(self, batch_n, doc_l, device='cpu'):
        # self.sent_hidden作为nn.LSTM()的第二个输入
        # 两个元组，shape都是(2,batch_n*doc_l,hidden_dim)
        # batch_n*doc_l相当于句子数量，
        # torch.rand()均匀分布，从区间[0, 1)的均匀分布中抽取的一组随机数
        # uniform_(),将tensor从均匀分布中抽样数值进行填充
        self.sent_hidden = (torch.rand(2, batch_n * doc_l, self.hidden_dim, device=device).uniform_(-0.01, 0.01),
                            torch.rand(2, batch_n * doc_l, self.hidden_dim, device=device).uniform_(-0.01, 0.01))
        # 两个元组，shape都是(2,batch_n,hidden_dim)
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
        sentpres = sentpres.transpose(0, 1)  # sentpres: (doc_l, batch_n, hidden_dim*2)

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
        name = 'st_rs_spp%s' % self.pool_type[0]
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