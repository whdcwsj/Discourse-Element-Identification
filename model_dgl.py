import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dgltor

from subLayer import *

# pytorch里面自定义层也是通过继承自nn.Module类来实现的
# pytorch里面一般是没有层的概念，层也是当成一个模型来处理的
class GraphConvLayer(nn.Module):
    """
    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    weight : bool, optional
        If True, apply a linear layer. Otherwise, aggregating the messages
        without a weight matrix.
    bias : bool, optional
        True if bias is added. Default: True
    activation : callable, optional
        Activation function. Default: None
    dropout : float, optional
        Dropout rate. Default: 0.0

    GraphConv → Parameters
    ----------
    allow_zero_in_degree (bool, optional)
        If there are 0-in-degree nodes in the graph, output for those nodes will be invalid since no message will be passed to those nodes.
        This is harmful for some applications causing silent performance regression.
        This module will raise a DGLError if it detects 0-in-degree nodes in input graph.
        By setting True, it will suppress the check and let the users handle it by themselves. Default: False.
    """

    def __init__(self, in_feat, out_feat, weight=True, bias=True, activation=None, dropout=0.0):
        super(GraphConvLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat

        self.conv = dgltor.GraphConv(in_feat, out_feat, norm='both', weight=weight, bias=bias, activation=activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, inputs):
        hs = self.conv(g, inputs)
        hs = self.dropout(hs)
        return hs



# 针对中文数据集
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

        self.SAGE_GCN = nn.ModuleList(
            [dgltor.SAGEConv(self.sent_dim * 2, self.sent_dim * 2, 'gcn', feat_drop=0.1, activation=nn.ReLU())
             for _ in range(dgl_layer)])

        self.temp_layer = nn.Sequential(
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

    def build_graph(self, sentence_encoding):
        # 构建图
        nodes_num = len(sentence_encoding)
        edges = []
        # 计算边的权重
        edges_weight = []
        # 构建图
        for i in range(nodes_num):
            for j in range(nodes_num):
                edges.append((i, j))
                # ------------------------------------------------------------
                # 计算余弦相似度
                weight = torch.cosine_similarity(sentence_encoding[i], sentence_encoding[j], dim=0)
                # ------------------------------------------------------------
                # 计算欧式距离的相似度
                # distance = torch.pairwise_distance(sent_encoding[i][None, :],
                #                                    sent_encoding[j][None, :])
                # weight = 1 / (1 + distance[0])
                # ------------------------------------------------------------
                # 计算pearson相关性系数
                # pearson = np.corrcoef(sent_encoding[i].cpu().detach().numpy(),
                #                       sent_encoding[j].cpu().detach().numpy())[0, 1]
                # weight = torch.tensor(pearson).to(torch.float32).to(self.config.device)
                # ------------------------------------------------------------
                # 计算kendall系数
                # kendall = pd.Series(sent_encoding[i].cpu().detach().numpy()).corr(
                #     pd.Series(sent_encoding[j].cpu().detach().numpy()), method="kendall")
                # weight = torch.tensor(kendall).to(torch.float32).to(self.config.device)
                # ------------------------------------------------------------
                edges_weight.append(weight)
                # 额外添加一个自循环
                if i == j:
                    edges.append((i, j))
                    edges_weight.append(weight)
        edges = torch.tensor(edges)
        graph = dgl.graph((edges[:, 0], edges[:, 1])).to(self.config.device)
        # 额外添加一个自循环
        # graph = dgl.add_self_loop(graph)
        edges_weight = torch.tensor(edges_weight).to(self.config.device)
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
        sentpres = self.posLayer(sentpres, pos)  # sentpres:(batch_n, doc_l, hidden_dim*2)


        # 可以加dgl的位置1(先加dgl，再加pos)


        sentpres = sentpres.transpose(0, 1)  # sentpres: (doc_l, batch_n, hidden_dim*2)

        tag_out, _ = self.tagLayer(sentpres, self.tag_hidden)  # tag_out: (doc_l, batch_n, sent_dim*2)
        # tag_out = self.dropout(tag_out)

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
        sentence_feature = []
        for i in range(len(length_essay)):
            inner_sentennce = tag_out[i]
            # build graph
            graph, edge_weight = self.build_graph(inner_sentennce)

            # add different GCN，句间交互
            for gcn in self.sage_gcns:
                inner_sentennce = gcn(graph, inner_sentennce, edge_weight)
                sentence_feature.append(inner_sentennce)

        # feature_bank = torch.cat(feature_bank, dim=-1)
        # inner_pred = self.middle_layer(feature_bank)[None, :, :]

        # ----------------------------------------------------------------------






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