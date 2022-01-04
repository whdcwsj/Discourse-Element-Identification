import torch
import torch.nn as nn
import torch.nn.functional as F

from subLayer import *


# 无句子间特征模型
# 少了pool_type
# 只添加了句子位置编码与过了LST妈的document进行加和，没有使用两个句间注意力
class STModel(nn.Module):
    def __init__(self, word_dim, hidden_dim, sent_dim, class_n, p_embd=None, pos_dim=0, p_embd_dim=16):
        # p_embd: 'cat', 'add','embd', 'embd_a'
        super(STModel, self).__init__()
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.sent_dim = sent_dim
        self.class_n = class_n
        self.p_embd = p_embd
        self.p_embd_dim = p_embd_dim

        # self.dropout = nn.Dropout(p=0.1)

        self.sentLayer = nn.LSTM(self.word_dim, self.hidden_dim, bidirectional=True)
        self.classifier = nn.Linear(self.sent_dim * 2, self.class_n)
        # self.classifier2 = nn.Linear(self.sent_dim, self.class_n)

        self.posLayer = PositionLayer(p_embd, p_embd_dim)

        if p_embd == 'embd':
            self.tagLayer = nn.LSTM(self.hidden_dim * 2 + p_embd_dim * 3, self.sent_dim, bidirectional=True)
        elif p_embd == 'cat':
            self.tagLayer = nn.LSTM(self.hidden_dim * 2 + 3, self.sent_dim, bidirectional=True)
        else:
            self.tagLayer = nn.LSTM(self.hidden_dim * 2, self.sent_dim, bidirectional=True)

    def init_hidden(self, batch_n, doc_l, device='cpu'):
        self.sent_hidden = (torch.rand(2, batch_n * doc_l, self.hidden_dim, device=device).uniform_(-0.01, 0.01),
                            torch.rand(2, batch_n * doc_l, self.hidden_dim, device=device).uniform_(-0.01, 0.01))
        self.tag_hidden = (torch.rand(2, batch_n, self.sent_dim, device=device).uniform_(-0.01, 0.01),
                           torch.rand(2, batch_n, self.sent_dim, device=device).uniform_(-0.01, 0.01))

    def forward(self, documents, pos=None, device='cpu', mask=None):
        batch_n, doc_l, sen_l, _ = documents.size()  # documents: (batch_n, doc_l, sen_l, word_dim)
        self.init_hidden(batch_n=batch_n, doc_l=doc_l, device=device)
        documents = documents.view(batch_n * doc_l, sen_l, -1).transpose(0,
                                                                         1)  # documents: (sen_l, batch_n*doc_l, word_dim)

        sent_out, _ = self.sentLayer(documents, self.sent_hidden)  # sent_out: (sen_l, batch_n*doc_l, hidden_dim*2)

        if mask is None:
            sentpres = torch.tanh(torch.mean(sent_out, dim=0))  # sentpres: (batch_n*doc_l, hidden_dim*2)
        else:
            sent_out = sent_out.masked_fill(mask.transpose(1, 0).unsqueeze(-1).expand_as(sent_out), 0)
            sentpres = torch.tanh(torch.sum(sent_out, dim=0) / (sen_l - mask.sum(dim=1).float() + 1e-9).unsqueeze(-1))

        # sentpres = torch.tanh(sent_out[-1])     # sentpres: (batch_n*doc_l, hidden_dim*2)

        sentpres = sentpres.view(batch_n, doc_l, self.hidden_dim * 2)  # sentpres: (batch_n, doc_l, hidden_dim*2)
        # sentpres = self.dropout(sentpres)
        # self.sent1 = sentpres   

        sentpres = self.posLayer(sentpres, pos)
        # self.sent2 = sentpres

        sentpres = sentpres.transpose(0, 1)

        tag_out, _ = self.tagLayer(sentpres, self.tag_hidden)  # tag_out: (tag_out, batch_n, output_dim*2)
        tag_out = torch.tanh(tag_out)
        # self.sent3 = tag_out

        tag_out = tag_out.transpose(0, 1)  # tag_out: (batch_n, doc_l, sent_dim*2)

        result = self.classifier(tag_out)

        result = F.log_softmax(result, dim=2)  # result: (batch_n, doc_l, class_n)
        return result

    def getModelName(self):
        name = 'st'
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


# type_id=0
# 用于中文数据集的情况下：
# 输入：vector_size=200，hidden_size=128，sent_dim=128，class_n=8，p_embd='add'，p_embd_dim=16，pool_type='max_pool'
# 英文数据集：
# p_embd = None
class STWithRSbySPP(nn.Module):
    def __init__(self, word_dim, hidden_dim, sent_dim, class_n, p_embd=None, pos_dim=0, p_embd_dim=16,
                 pool_type='max_pool'):
        # p_embd: 'cat', 'add','embd', 'embd_a'
        super(STWithRSbySPP, self).__init__()
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

        # 双向nn.LSTM()
        # input：(40,batch_n*doc_l,200) → (seq_len, batch, input_size)
        # (h_0,c_0)：h_0是隐藏层的初始状态，c_0是初始化的细胞状态
        # 这两者shape均为: (num_layers * num_directions, batch, hidden_size)
        # 输出：output,(h_n,c_n)
        # output保存RNN最后一层的输出的Tensor，(seq_len, batch, hidden_size * num_directions) → (40,batch_n*doc_l,hidden_dim*2)
        # (h_n，c_n):保存着RNN最后一个时间步的隐藏层状态；保存着RNN最后一个时间步的细胞状态
        # shape：(num_layers * num_directions, batch, hidden_size)
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

        # LSTM(self.hidden_dim*2, self.sent_dim, bidirectional=True)
        # input：(seq_len, batch, input_size)，input_size应该=LSTM的第一个参数，也就是self.hidden_dim*2
        # (h_0,c_0)：(num_layers * num_directions, batch, hidden_size)
        # output: (seq_len, batch, sent_dim * num_directions)
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




# 尝试相对位置编码不同的权重赋值
class STWithRSbySPP_Weight(nn.Module):
    def __init__(self, word_dim, hidden_dim, sent_dim, class_n, p_embd=None, pos_dim=0, p_embd_dim=16,
                 pool_type='max_pool', weight_matrix=None):
        # p_embd: 'cat', 'add','embd', 'embd_a'
        super(STWithRSbySPP_Weight, self).__init__()
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.sent_dim = sent_dim
        self.class_n = class_n
        self.p_embd = p_embd
        self.p_embd_dim = p_embd_dim
        self.pool_type = pool_type
        self.weight_matrix = weight_matrix

        self.dropout = nn.Dropout(0.1)
        self.sentLayer = nn.LSTM(self.word_dim, self.hidden_dim, bidirectional=True)
        # 为什么sent_dim*2+30？？？因为要接两个句间注意力
        self.classifier = nn.Linear(self.sent_dim * 2 + 30, self.class_n)

        # 配合avg与max加和时进行使用
        # self.classifier = nn.Linear(self.sent_dim * 2 + 60, self.class_n)

        self.posLayer = PositionLayer(p_embd, p_embd_dim, weight_matrix=self.weight_matrix)

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
        if self.weight_matrix is not None:
            sentpres = self.posLayer(sentpres, pos)
        else:
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






# 单纯的英文模型采用LSTM+SPP的两个dropout0.1效果好一些
# type_id=1
class EnSTWithRSbySPP(nn.Module):
    def __init__(self, word_dim, hidden_dim, sent_dim, class_n, p_embd=None, pos_dim=0, p_embd_dim=16,
                 pool_type='max_pool'):
        # p_embd: 'cat', 'add','embd', 'embd_a'
        super(EnSTWithRSbySPP, self).__init__()
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

        # 双向nn.LSTM()
        # input：(40,batch_n*doc_l,200) → (seq_len, batch, input_size)
        # (h_0,c_0)：h_0是隐藏层的初始状态，c_0是初始化的细胞状态
        # 这两者shape均为: (num_layers * num_directions, batch, hidden_size)
        # 输出：output,(h_n,c_n)
        # output保存RNN最后一层的输出的Tensor，(seq_len, batch, hidden_size * num_directions) → (40,batch_n*doc_l,hidden_dim*2)
        # (h_n，c_n):保存着RNN最后一个时间步的隐藏层状态；保存着RNN最后一个时间步的细胞状态
        # shape：(num_layers * num_directions, batch, hidden_size)
        sent_out, _ = self.sentLayer(documents, self.sent_hidden)  # sent_out: (sen_l, batch_n*doc_l, hidden_dim*2)

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
        sentFt = self.dropout(sentFt)

        # "add"情况下，将前三个pos位置1：1：1与sentence加和; ['gpos', 'lpos', 'ppos']
        sentpres = self.posLayer(sentpres, pos)  # sentpres:(batch_n, doc_l, hidden_dim*2)
        sentpres = sentpres.transpose(0, 1)  # sentpres: (doc_l, batch_n, hidden_dim*2)

        # LSTM(self.hidden_dim*2, self.sent_dim, bidirectional=True)
        # input：(seq_len, batch, input_size)
        # (h_0,c_0)：(num_layers * num_directions, batch, hidden_size)
        # output: (seq_len, batch, sent_dim * num_directions)
        tag_out, _ = self.tagLayer(sentpres, self.tag_hidden)  # tag_out: (doc_l, batch_n, sent_dim*2)

        tag_out = torch.tanh(tag_out)

        tag_out = tag_out.transpose(0, 1)  # tag_out: (batch_n, doc_l, sent_dim*2)
        roleFt = self.rfLayer(tag_out)  # roleFt:(batch_n, doc_l, 15)
        roleFt = self.dropout(roleFt)

        tag_out = torch.cat((tag_out, sentFt, roleFt), dim=2)  # tag_out: (batch_n, doc_l, sent_dim*2+30)  (1,8,286)

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




# type_id=2
# 多了ft_size
# sentence+额外的feature
class STWithRSbySPPWithFt2(nn.Module):
    def __init__(self, word_dim, hidden_dim, sent_dim, class_n, p_embd=None, pos_dim=0, p_embd_dim=16, ft_size=0,
                 pool_type='max_pool'):
        # p_embd: 'cat', 'add','embd', 'embd_a'
        super(STWithRSbySPPWithFt2, self).__init__()
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.sent_dim = sent_dim
        self.class_n = class_n
        self.p_embd = p_embd
        self.p_embd_dim = p_embd_dim
        self.ft_size = ft_size
        self.pool_type = pool_type

        self.dropout = nn.Dropout(0.5)
        self.sentLayer = nn.LSTM(self.word_dim, self.hidden_dim, bidirectional=True)
        self.classifier = nn.Linear(self.sent_dim * 2 + 30, self.class_n)

        self.posLayer = PositionLayer(p_embd, p_embd_dim)
        self.sfLayer = InterSentenceSPPLayer(self.hidden_dim * 2, pool_type=self.pool_type)
        self.rfLayer = InterSentenceSPPLayer(self.hidden_dim * 2, pool_type=self.pool_type)

        if p_embd == 'embd':
            self.tagLayer = nn.LSTM(self.hidden_dim * 2 + p_embd_dim * 3 + ft_size, self.sent_dim, bidirectional=True)
        elif p_embd == 'cat':
            self.tagLayer = nn.LSTM(self.hidden_dim * 2 + 3 + ft_size, self.sent_dim, bidirectional=True)
        else:
            self.tagLayer = nn.LSTM(self.hidden_dim * 2 + ft_size, self.sent_dim, bidirectional=True)

    def init_hidden(self, batch_n, doc_l, device='cpu'):
        self.sent_hidden = (torch.rand(2, batch_n * doc_l, self.hidden_dim, device=device).uniform_(-0.01, 0.01),
                            torch.rand(2, batch_n * doc_l, self.hidden_dim, device=device).uniform_(-0.01, 0.01))
        self.tag_hidden = (torch.rand(2, batch_n, self.sent_dim, device=device).uniform_(-0.01, 0.01),
                           torch.rand(2, batch_n, self.sent_dim, device=device).uniform_(-0.01, 0.01))

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
        name = 'sent_SPP_%s_ft_%d' % (self.pool_type[0], self.ft_size)
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
