import torch
import torch.nn as nn
import torch.nn.functional as F

from subLayer import *
from src.config import Config
from transformers import BertModel


class OriginalBertClassification(nn.Module):
    def __init__(self, config: Config, bert_trainable=False):
        """
        :param word_dim: 768
        :param hidden_dim: 128
        :param sent_dim: 128
        :param class_n: 8
        :param p_embd: 'add' ; 若是['embd_b', 'embd_c', 'addv']
        :param p_embd_dim: 16 ; 则hidden_dim * 2
        :param pool_type: 'max_pool'
        """
        # p_embd: 'cat', 'add','embd', 'embd_a'
        super(OriginalBertClassification, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config.bert_path)
        # bert模型的参数是否冻结
        for param in self.bert.parameters():
            param.requires_grad = bert_trainable

        self.word_dim = self.config.word_dim
        self.hidden_dim = self.config.hidden_dim
        self.sent_dim = self.config.sent_dim
        self.class_n = self.config.class_n
        self.p_embd = self.config.p_embd
        self.p_embd_dim = self.config.p_embd_dim
        self.pool_type = self.config.pool_type

        # self.dropout = nn.Dropout(0.1)
        self.sentLayer = nn.LSTM(self.word_dim, self.hidden_dim, num_layers=1, bidirectional=self.config.bidirectional)
        # 为什么sent_dim*2+30？？？因为要接两个句间注意力
        self.classifier = nn.Linear(self.sent_dim * 2 + 30, self.class_n)

        # 配合avg与max加和时进行使用
        # self.classifier = nn.Linear(self.sent_dim * 2 + 60, self.class_n)

        self.posLayer = PositionLayer(self.p_embd, self.p_embd_dim)

        self.sfLayer = InterSentenceSPPLayer(self.hidden_dim * 2, pool_type=self.pool_type)
        self.rfLayer = InterSentenceSPPLayer(self.hidden_dim * 2, pool_type=self.pool_type)

        # 配合avg与max加和时进行使用
        # self.sfLayer = InterSentenceSPPLayer3(self.hidden_dim*2, pool_type = self.pool_type)
        # self.rfLayer = InterSentenceSPPLayer3(self.hidden_dim*2, pool_type = self.pool_type)

        # 这个是将三个位置编码torch.cat到了一起，所以input_size要再加上 p_embd_dim*3
        if self.p_embd == 'embd':
            self.tagLayer = nn.LSTM(self.hidden_dim * 2 + self.p_embd_dim * 3, self.sent_dim, num_layers=1,
                                    bidirectional=self.config.bidirectional)
        # 这个是将三个位置编码torch.cat到了一起，所以input_size要再加上3
        elif self.p_embd == 'cat':
            self.tagLayer = nn.LSTM(self.hidden_dim * 2 + 3, self.sent_dim, num_layers=1,
                                    bidirectional=self.config.bidirectional)
        else:
            self.tagLayer = nn.LSTM(self.hidden_dim * 2, self.sent_dim, num_layers=1,
                                    bidirectional=self.config.bidirectional)

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

    # 输入：
    # document:(1,batch_n,doc_l,sen_l)
    # pos:(batch_n,doc_l,6)
    # mask:NONE
    def forward(self, documents, pos=None, mask=None):
        documents = documents.squeeze(0)

        # documents的可能需要先放在CPU上加载
        temp_batch_output = []
        for i in range(documents.shape[0]):
            embedding = self.bert(documents[i])
            last_hidden_state = embedding[0]
            last_hidden_state = last_hidden_state.to(self.config.device)
            temp_batch_output.append(last_hidden_state)

        batch_bert_output = torch.stack(temp_batch_output, dim=0)

        batch_n, doc_l, sen_l, _ = batch_bert_output.size()  # documents: (batch_n, doc_l, sen_l, word_dim)
        self.init_hidden(batch_n=batch_n, doc_l=doc_l, device=self.config.device)
        documents = batch_bert_output.view(batch_n * doc_l, sen_l, -1).transpose(0,1)  # documents: (sen_l, batch_n*doc_l, word_dim)


        sent_out, _ = self.sentLayer(documents, self.sent_hidden)  # sent_out: (sen_l, batch_n*doc_l, hidden_dim*2)
        # sent_out = self.dropout(sent_out)

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
        pos = pos.squeeze(0)
        sentpres = self.posLayer(sentpres, pos)  # sentpres:(batch_n, doc_l, hidden_dim*2)
        sentpres = sentpres.transpose(0, 1)  # sentpres: (doc_l, batch_n, hidden_dim*2)


        tag_out, _ = self.tagLayer(sentpres, self.tag_hidden)  # tag_out: (doc_l, batch_n, sent_dim*2)
        # tag_out = self.dropout(tag_out)

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