import dgl
import dgl.nn.pytorch as dgltor
import torch
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
# g.ndata['y'] = th.randn(g.num_nodes(), 5)
#
# print(g)
# print(g.ndata['x'][1])
# print(g.ndata['y'][1])
# print(g.edata['x'][th.tensor([0, 3])])  # 获取边0和3的特征
#
# bg1 = dgl.to_bidirected(g, copy_ndata=True)
# print(bg1)




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


# 4、各种节点间相似度测试
# t1 = th.tensor([0.1, 0.2, 0.3, 0.55])
# t1 = th.tensor([0., 0., 0., 0.])
# t2 = th.tensor([0.3, 0.21, 0.56, 0.8])
# # t1 * t2 = 0.68
# # 计算cos相似度
# print(th.cosine_similarity(t1, t2, dim=0))
# # 计算欧式距离
# print(th.pairwise_distance(t1[None, :], t2[None, :])[0])
# 计算pearson系数
# print(np.corrcoef(t1, t2)[0, 1])
# # 计算kendall系数
# t1 = pd.Series(t1)
# print(t1.corr(pd.Series(t2), method="kendall"))




# # 5、如何通过节点对添加多个边
# g = dgl.DGLGraph()
# g.add_nodes(34)
# edge_list = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2),
#              (4, 0), (5, 0), (6, 0), (6, 4), (6, 5), (7, 0), (7, 1),
#              (7, 2), (7, 3), (8, 0), (8, 2), (9, 2), (10, 0), (10, 4),
#              (10, 5), (11, 0), (12, 0), (12, 3), (13, 0), (13, 1), (13, 2),
#              (13, 3), (16, 5), (16, 6), (17, 0), (17, 1), (19, 0), (19, 1),
#              (21, 0), (21, 1), (25, 23), (25, 24), (27, 2), (27, 23),
#              (27, 24), (28, 2), (29, 23), (29, 26), (30, 1), (30, 8),
#              (31, 0), (31, 24), (31, 25), (31, 28), (32, 2), (32, 8),
#              (32, 14), (32, 15), (32, 18), (32, 20), (32, 22), (32, 23),
#              (32, 29), (32, 30), (32, 31), (33, 8), (33, 9), (33, 13),
#              (33, 14), (33, 15), (33, 18), (33, 19), (33, 20), (33, 22),
#              (33, 23), (33, 26), (33, 27), (33, 28), (33, 29), (33, 30),
#              (33, 31), (33, 32)]
# # *的作用相当于解压 zip(*list) = zip(晒出list中的各元素)
# # zip()会直接打包为元祖的列表
# src, dst = tuple(zip(*edge_list))
# g.add_edges(src, dst)
# g.add_edges(dst, src)
#
# print('We have %d nodes.' % g.number_of_nodes())
# print('We have %d edges.' % g.number_of_edges())
#
# nx_G = g.to_networkx().to_undirected()
# # Kamada-Kawaii layout usually looks pretty for arbitrary graphs 任意图
# pos = nx.kamada_kawai_layout(nx_G)
# nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .230, .7]])
# plt.show()



# 6、节点相似度
# input1 = th.randn(2,3)
# print(input1)
#
# in1=input1[0:1, :]
# in2=input1[1:2, :]
# print(th.cosine_similarity(in1, in2))

# t3=th.tensor([[3, 8, 7, 5, 2, 9], [3, 8, 7, 5, 2, 9]], dtype=torch.float64)
# t4=th.tensor([[1, 8, 6, 6, 4, 5], [3, 8, 7, 5, 2, 9]], dtype=torch.float64)
# t5=th.tensor([[1, 8, 6, 6, 4, 5], [3, 8, 7, 5, 2, 9]], dtype=torch.float64)
# print(t3.shape)
# wang= []
# wang.append(t3)
# wang.append(t4)
# wang.append(t5)t
# wang=torch.stack(wang, dim=0)
# print(wang)
# print(wang.shape)





# input1 = torch.randn(100, 128)
# input2 = torch.randn(100, 128)
# output = th.cosine_similarity(input1, input2)
# print(output)
# print(output.shape)


# 7、图中边的添加
# edges = []
# edges.append((1,2))
# edges.append((3,5))
# edges.append((7,8))
# print(edges)
# edges = torch.tensor(edges)
# print(edges.shape)
#
# print((edges[:, 0], edges[:, 1]))


# 8、SAGEConv的输出值测试
# g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
# g = dgl.add_self_loop(g)
# print(g)  # 六个节点，十二条边
# feat = th.ones(6, 10)
# conv = dgltor.SAGEConv(10, 2, 'pool')
# res = conv(g, feat)
# print(res)


# 9、节点间的相似度计算方法
# ------------------------------------------------------------
# 余弦相似度[-1,1]
print("余弦")
t3=th.tensor([3, 8, 7, 5, 2, 9, 6], dtype=torch.float)
t4=th.tensor([1, 8, 6, 6, 4, 5, 6], dtype=torch.float)
t5=th.tensor([1, 8, 6, 6, 4, 5, 6], dtype=torch.float)
print(t3.shape)
print(th.cosine_similarity(t3, t4, dim=0))
print(th.cosine_similarity(t5, t4, dim=0))
print("------------------------------------------------------------")
# Pearson 相似度（Pearson Similarity）[-1,1]
print("pearson相似度")
pearson1 = np.corrcoef(t3.cpu().detach().numpy(), t4.cpu().detach().numpy())[0][1]
pearson2 = np.corrcoef(t5.cpu().detach().numpy(), t4.cpu().detach().numpy())[0][1]
weight1 = torch.tensor(pearson1, dtype=torch.float)
weight2 = torch.tensor(pearson2, dtype=torch.float)
print(weight1)
print(weight2)
print("------------------------------------------------------------")
# 计算欧式距离（Euclidean Distance）的相似度
print("欧氏距离")
# 输入:(N,D)其中D等于向量的维度
# 输出:(N,1)
# 距离越大，权重越小，作为分母的时候，补一个1
distance1 = th.pairwise_distance(t3[None, :], t4[None, :])
distance2 = th.pairwise_distance(t5[None, :], t4[None, :])
print(distance1[0])
print(distance2[0])
weight1 = 1 / (1 + distance1[0])
weight2 = 1 / (1 + distance2[0])
print(weight1)
print(weight2)
print("------------------------------------------------------------")
# 统计学三大相关系数
# 计算kendall系数
print("kendall系数")
kendall1 = pd.Series(t3.cpu().detach().numpy()).corr(pd.Series(t4.cpu().detach().numpy()), method="kendall")
kendall2 = pd.Series(t5.cpu().detach().numpy()).corr(pd.Series(t4.cpu().detach().numpy()), method="kendall")
weight1 = torch.tensor(kendall1)
weight2 = torch.tensor(kendall2)
print(weight1)
print(weight2)

# Dice这种求交、并、补、差的集合，暂时不纳入考虑
