import json
import random
import math
import dgl
import dgl.nn.pytorch as dgltor
import torch

# jsonData = '{"a":1,"b":2,"c":3,"d":4,"e":"wsj"}'
# text = json.loads(jsonData)
# print(text)

# wang=[]
# list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# title = False
# wang.append(list[:5+title])
# print(wang)


# list1 = ["Thesis", "Elaboration"]
# for i in list1:
#     for j in i:
#         print(j)
#         break

# import torch
# import torch.nn as nn
# a=torch.ones(2,3,4)
# a[0,1,2:4]=0
# print(a)
#
# avg1=nn.AdaptiveAvgPool1d(5)
# avg2=nn.AdaptiveAvgPool1d(2)
#
# print(avg1(a))
# print(avg2(a))

# pool_type='max_pool'
# print(pool_type[0])

# arr=[[4,5],[4,6],[6,3],[2,3],[1,1]]
# arr.sort(key=lambda x:x[0]) #statement 1
# print(arr)
# # [[1, 1], [2, 3], [4, 5], [4, 6], [6, 3]]
# # the first element, negative of the second element
# arr.sort(key=lambda x:(x[0],-x[1])) #statement 2
# print(arr)
# # [[1, 1], [2, 3], [4, 6], [4, 5], [6, 3]]

# import torch
# a=torch.rand(2,8).uniform_(-8, 8)
# print(a)
# b=torch.Tensor(2,3).uniform_(-1,1)
# print(b)

# import torch
# x = torch.Tensor([1, 2, 3, 4, 5, 6, 8, 2, 3, 4, 5, 6]).view(2, 2, 3)
# print(x)
# print(x.shape)
# y_0=torch.mean(x,dim=2)
# print(y_0)
# print(y_0.shape)

# import torch
# x = torch.Tensor([1, 2, 3, 4, 5, 6, 8, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 8, 2, 3, 4, 5, 6]).view(2, 3, 4)
# print(x)
# print(x.shape)
# z = x.transpose(2, 1)
# print(z.shape)

# import torch
# import torch.nn as nn
# m = nn.AdaptiveMaxPool1d(15)
# input = torch.randn(1, 8, 8)
# output = m(input)
# print(output.size())


# x = torch.Tensor(2, 3)
# # y = x.permute(1,0)
# # y.view(-1)
# y = x.permute(1, 0).contiguous()
# print(y.shape)
# z = y.view(-1)
# print(z.shape)

# b = torch.Tensor([1, 2, 3, 4, 5, 6])
# print(b)
# print(b.shape)
# c = b.view(6)
# print(c)
# print(c.shape)

# import pickle

# embeddings = torch.load('./embd/bert-base-uncased-word_embeddings.pkl')
# print(embeddings.embedding_dim)
# print(embeddings)

# wang = [1,2,3,4,5,6,7,8]
# print(wang[:-1])

# from transformers import BertTokenizer
#
# first_person_list = ['I', 'me', 'my', 'mine', 'myself']
#
# BERT_PATH = '/home/wsj/bert_model/bert-base-uncased'
# tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
# en_fp_list = tokenizer.encode(' '.join(first_person_list), add_special_tokens=True)
# print(en_fp_list)

# [1045, 2033, 2026, 3067, 2870]

# [101, 1045, 2033, 2026, 3067, 2870, 102]


# import torch
# import torch.nn as nn
# print(torch.arange(0., 100).shape)
# position = torch.arange(0., 100).unsqueeze(1)
# print(position.shape)
# # print(position)
# pe = torch.zeros(100,16)
# print(pe.shape)
# print(pe[:, 0::2].shape)
# print(pe[:, 0::2])

# wang = torch.zeros(100, 16, 3)
# print(wang.shape)
# si = wang[ : , : , 0]
# print(si.shape)

# import torch
# import torch.nn as nn
# from torch import autograd

# m = nn.Softmax()
# input = autograd.Variable(torch.randn(2, 3))
# print(input)
# print(m(input))

# y = torch.tensor([[[1.,2.,3.],[4.,5.,6.]],[[7.,8.,9.],[10.,11.,12.]]])
# print(y.shape)
# #y的size是2,2,3。可以看成有两张表，每张表2行3列
# net_1 = nn.Softmax(dim=0)
# net_2 = nn.Softmax(dim=1)
# net_3 = nn.Softmax(dim=2)
# print('dim=0的结果是：\n',net_1(y),"\n")
# print('dim=1的结果是：\n',net_2(y),"\n")
# print(net_2(y).shape)
# print('dim=2的结果是：\n',net_3(y),"\n")
# print(net_3(y).shape)

# print(torch.cuda.is_available())

# import torch
#
# a = [[[1,1,1,1],[2,2,2,2],[3,3,3,3]],[[1,1,1,1],[2,2,2,2],[3,3,3,3]]]
# a = torch.tensor(a)
# print(a)
# print(a.shape)
#
# b =[[0.3,0.4,0.5],[0.1,0.2,0.9]]
# b = torch.tensor(b)
# print(b)
# print(b.shape)
# #
# # # # 这样会导致竖着乘积
# # # # c = a*b
# # # # print(c)
# #
# # 这样横着乘积才对
# print(a*b.unsqueeze(-1))
# d = torch.sum(a*b.unsqueeze(-1), dim=1)
# print(d)
# print(d.shape)

# x = torch.randn(2, 3)#为1可以扩展为3和4
# print(x)
# x = x.expand(2, 3, 1)
# print(x)
# print(x.shape)


# b = 1
# a = [1,2,3,4,5,6]
# if b in a:
#     print("www")
# else:
#     print("nnn")


# append_idx = random.sample(range(0, 3), int(2))
# print(append_idx)

# num = 0
# for i in range(0, 1001, 100):
#     print(111)
#     num += 1
# print(num)


# num = random.randint(0,1)
# print(num)

# wang = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# length = len(wang)
# print(type(length))
#
# num = int(math.ceil(length/3))
# print(num)
# print(type(num))

# for i in range(10):
#     if i<5:
#         print(111)
#     else:
#         break

# wang = 0
# if wang:
#     wang = True
# else:
#     wang = False
# print(wang)

# num = 3
# wang = [1,2,3]
# si = [4,5,6,7]
#
# print(wang+si)

# wang = 0.1
# assert wang in [0, 1]
# print(wang)

# g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
# g = dgl.add_self_loop(g)
# feat = torch.ones(6, 10)
# gatconv = dgltor.GATConv(10, 2, num_heads=5)
# res = gatconv(g, feat)
# print(res.shape)
# print(res)
# print("------------")

# wang = []
# for i in range(1):
#     print(res[:, i:i+1, :])
#     temp = res[:, i:i+1, :].squeeze(dim=1)
#     print(temp)
#     # print(temp.shape)
#     wang.append(temp)
# # print(wang)
#
# wang = torch.stack(wang)
# print(wang.shape)


# inner_sentennce = torch.mean(res, dim=1)
# print(inner_sentennce.shape)
# print(inner_sentennce)

wang = [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
print(wang[:11])