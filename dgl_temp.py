import dgl
import torch as th
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# # 1、同构图
# g = dgl.graph(([0, 0, 1, 5], [1, 2, 2, 0]))
# print(g)
#
# weights = th.tensor([0.1, 0.6, 0.9, 0.7])  # 每条边的权重
# g.edata['w'] = weights  # 将其命名为 'w'
# g.ndata['x'] = th.ones(g.num_nodes(), 3)
# g.edata['x'] = th.ones(g.num_edges(), dtype=th.int32)
# print(g)
#
# g.ndata['y'] = th.randn(g.num_nodes(), 5)
# print(g)
# print(g.ndata['x'][1])
# print(g.ndata['y'][1])
# print(g.edata['x'][th.tensor([0, 3])])  # 获取边0和3的特征
#
# # 同构图的可视化(异构图不行)
# nx_g = g.to_networkx().to_undirected()
# pos = nx.kamada_kawai_layout(nx_g)
# nx.draw(nx_g, pos, with_labels=True, node_color=[[.7, .7, .7]])
# plt.show()


# # 2、异构图
# 创建一个具有3种节点类型和3种边类型的异构图
# 下面的代码构造了一个包含“药物”、“基因”、“疾病”三种顶点和“反应”、“作用”、“治疗”三种边的异构图
# (源节点类型, 边类型, 目标节点类型)
# graph_data = {
#     # 代表两条边
#     ('drug', 'interacts', 'drug'): (th.tensor([0, 1]), th.tensor([1, 2])),
#     ('drug', 'interacts', 'gene'): (th.tensor([0, 1]), th.tensor([2, 3])),
#     ('drug', 'treats', 'disease'): (th.tensor([1]), th.tensor([2]))
# }
# g = dgl.heterograph(graph_data)
# print(g)
# print(g.ntypes)
# print(g.etypes)
# print(g.canonical_etypes)
# print("*******************")
# print(g.nodes('disease'))
# print(g.num_edges(('drug', 'treats', 'disease')))
# print(g.num_edges('treats'))
#
# g.nodes['drug'].data['hv'] = th.ones(g.num_nodes('drug'))
# print(g.nodes['drug'].data['hv'])
# g.edges['treats'].data['he'] = th.zeros(g.num_edges(('drug', 'treats', 'disease')))
# print(g.edges['treats'].data['he'])


# # 3、cpu/gpu设备测试
# u, v = th.tensor([0, 1, 2]), th.tensor([2, 3, 4])
# g = dgl.graph((u, v))
# g.ndata['x'] = th.randn(5, 3)
# print(g.device)
#
# cuda_g = g.to('cuda')
# print(cuda_g.device)
# print(cuda_g.ndata['x'].device)
#
# u, v = u.to('cuda'), v.to('cuda')
# dgl_g = dgl.graph((u, v))
# print(dgl_g.device)
# print("------------------")
#
# print(cuda_g.in_degrees())
# print(cuda_g.in_edges([2, 3]))
# cuda_g.ndata['h'] = th.randn(5,4).to('cuda')
# print(cuda_g.ndata['h'])


# # 4、各种节点间相似度测试
t1 = th.tensor([0.1, 0.2, 0.3, 0.55])
t2 = th.tensor([0.3, 0.21, 0.56, 0.8])
# t1 * t2 = 0.68
# 计算cos相似度
print(th.cosine_similarity(t1, t2, dim=0))
# 计算欧式距离
print(th.pairwise_distance(t1[None, :], t2[None, :])[0])
# 计算pearson系数
print(np.corrcoef(t1, t2)[0, 1])
# 计算kendall系数
t1 = pd.Series(t1)
print(t1.corr(pd.Series(t2), method="kendall"))





