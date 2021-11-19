import json

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

import torch

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

import pickle

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

print(torch.cuda.is_available())