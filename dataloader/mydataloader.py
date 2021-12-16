import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
from transformers import BertTokenizer
import json
from src.config import Config

# 句子长度不够时进行填充
PADDING = [0]

# 标签名称对应的ID号
label_map = {'Introduction': 0,
             'Thesis': 1,
             'Main Idea': 2,
             'Evidence': 3,
             'Conclusion': 4,
             'Other': 5,
             'Elaboration': 6,
             'padding': 7}


# 从绝对位置编码构建相对位置编码
# 输入类似：{'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 'wsj'}
def getRelativePos(load_dict):
    # 句子的全局位置
    gid = load_dict['gid']
    # 获取全局的相对位置，要除以|E|（句子数量）
    load_dict['gpos'] = [i/len(gid) for i in gid]
    # 句子的局部位置（每段中的第几句话）
    lid = load_dict['lid']
    lpos = []
    temp = []
    for i in lid:
        # 每当上一个段落序号统计结束时
        if i == 1 and len(temp) > 0 :
            lpos += [i/len(temp) for i in temp]
            temp = []
        temp.append(i)
    # 处理最后一个统计的段落
    if len(temp) > 0:
        lpos += [i/len(temp) for i in temp]
    # 获取局部的相对位置
    load_dict['lpos'] = lpos

    # 句子的段落位置
    pid = load_dict['pid']
    # 获取段落的相对位置
    load_dict['ppos'] = [i/pid[-1] for i in pid]
    return load_dict


# 返回每篇文章：句子列表(带titile)，每句话的标签列表，每个句子的按顺序对应的六个特征
# 输入：数据集，title=True
def loadDataAndFeature(in_file, title=False, max_len=99):
    labels = []
    documents = []
    ft_list = ['gpos', 'lpos', 'ppos', 'gid', 'lid', 'pid']
    features = []
    # 中文数据集每一行：file,title,score,sents,label,gid,lid,pid
    with open(in_file, 'r', encoding='utf-8') as f:
        for line in f:
            ft = []
            # 解码json数据，返回{'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 'wsj'}
            load_dict = json.loads(line)
            # 返回新的参数字典，获取相对位置['gpos', 'lpos', 'ppos']
            load_dict = getRelativePos(load_dict)
            # 数据中是否添加title
            if title:
                # if ('slen' in load_dict) and ('slen' not in ft_list):
                #     ft_list.append('slen')

                load_dict['sents'].insert(0, load_dict['title'])
                load_dict['labels'].insert(0, 'padding')
                for k in ft_list:
                    load_dict[k].insert(0, 0)

                # 记录title的长度
                # if 'slen' in load_dict:
                #     load_dict['slen'][0] = len(load_dict['title'])

            # 小于maxlen+title的内容都会包含进去
            # title：长度多1少1
            documents.append(load_dict['sents'][: max_len+title])
            labels.append(load_dict['labels'][: max_len+title])

            # ['gid']从1开始，一次性把每句话对应的六个指标都读进去
            for i in load_dict['gid']:
                if i > max_len:
                    break
                ft.append([load_dict[k][i-1+title] for k in ft_list])

            features.append(ft)

    return documents, labels, features


# 将label的文字形式转换为数字
def labelEncode(labels):
    en_labels = []
    for labs in labels:
        en_labs = []
        for label in labs:
            en_labs.append(label_map[label])
        en_labels.append(en_labs)
    return en_labels


# 每个句子的最大长度
def sentencePaddingId(documents, labels, n_l=40, is_cutoff=True):
    pad_documents = []
    # 每行
    for sentences in documents:
        length = len(sentences)
        out_sentences = []
        # 每个句子
        for sentence in sentences:
            if len(sentence) % n_l:
                sentence = sentence + PADDING * (n_l - len(sentence) % n_l)
            if is_cutoff:
                out_sentences.append(sentence[0: n_l])
        pad_documents.append(out_sentences)
    pad_labels = labels
    return pad_documents, pad_labels


class BertDataset(Dataset):
    def __init__(self, config, data_path):
        super(BertDataset, self).__init__()
        self.config = config
        self.data_list = []
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_path)

        # 返回每篇文章：句子列表(带titile)，每句话的标签列表，每个句子的按顺序对应的六个特征
        self.documents, labels, self.pos_features = loadDataAndFeature(data_path, title=True)

        # 所有文章，每片文章中所有句子的label，从文字转换为id
        self.id_labels = labelEncode(labels)


    def __getitem__(self, item):
        essay = self.documents[item]
        pos = self.pos_features[item]
        label = self.id_labels[item]

        # 保留了开头的[CLS]和结尾的[SEP]
        essay_id = self.tokenizer(essay)




        # tokenize
        # for sentence in sent:
        tokenized = self.tokenizer(sent, padding=True)
        token_ids = tokenized['input_ids']
        masks = tokenized['attention_mask']

        # 格式转换
        # pos_features = torch.tensor(pos_features, dtype=torch.float, device=config.device)[:, :, :6]
        # id_labels = torch.tensor(id_labels, dtype=torch.long, device=config.device)


        return essay, label, pos

    def __len__(self):
        return len(self)


# class MyDataset(Dataset):
#     def __init__(self, config, dataset, device, test=False):
#         super(MyDataset, self).__init__()
#         self.config = config
#         self.dataset = dataset
#         self.id_arr = np.asarray(self.dataset.iloc[:, 0])
#         self.text_arr = np.asarray(self.dataset.iloc[:, 1])
#         self.test = test
#         if self.test is False:
#             self.first_label_arr = np.asarray(self.dataset.iloc[:, 3])
#             self.second_label_arr = np.asarray(self.dataset.iloc[:, 4])
#         self.device = device
#         if config.embedding_pretrained_model is not None:
#             self.vob = config.embedding_pretrained_model.wv.key_to_index.keys()
#             # # 加入PAD 字符
#             # self.vob.append(0)
#
#     def __getitem__(self, item):
#         id_ = self.id_arr[item]
#         id_ = torch.tensor(id_).to(self.device)
#         token_ids = self.text_arr[item]
#         # 处理word embedding预训练的
#         if self.config.embedding_pretrained_model is not None:
#             token_ids_temp = []
#             # 如果不在word embedding中的token 则去掉
#             for index, token_id in enumerate(token_ids):
#                 if token_id in self.vob:
#                     # 用0号作为padding embedding多加入了一行， 所以index需要+1
#                     token_ids_temp.append(self.config.embedding_pretrained_model.wv.key_to_index[token_id] + 1)
#             token_ids = token_ids_temp
#         # padding and truncated
#         padding_len = self.config.pad_size - len(token_ids)
#         if padding_len >= 0:
#             token_ids = token_ids + [0] * padding_len
#         else:
#             token_ids = token_ids[:self.config.pad_size]
#         token_ids = torch.tensor(token_ids).to(self.device)
#         if self.test is False:
#             first_label = self.first_label_arr[item]
#             second_label = self.second_label_arr[item]
#             first_label = torch.tensor(first_label - 1).to(self.device)
#             second_label = torch.tensor(second_label - 1).to(self.device)
#             return token_ids, first_label, second_label
#         else:
#             return id_, token_ids
#
#     def __len__(self):
#         return len(self.id_arr)




if __name__ == '__main__':

    # config = Config(name='bert_classifier')
    # model = BertClassifier(config).to(config.device)
    # train_dataset = BertDataset(config, r'../data/train/train')
    # dataloader = DataLoader(train_dataset)
    # for data in dataloader:
    #     token_ids, masks, _, out, _ = data
    #     _output, _, _, _ = model(token_ids, masks)
    #     print()

    # documents, labels, features = loadDataAndFeature('../data/new_Ch_train.json', title=True)
    # print(len(documents))
    # print(len(labels))
    # print(len(features))


    tokenizer = BertTokenizer.from_pretrained(r'/home/wsj/bert_model/chinese/bert_chinese_L-12_H-768_A-12')
    text = '今天天气真不错'
    true_text = ["平凡", "的", "沙子", "中", "蕴含", "着", "宝贵", "的", "黄金", "，", "平凡", "的", "泥土", "里", "培养", "出", "鲜活", "的", "生命", "。"]
    new_text = ["平凡的沙子中蕴含着宝贵的黄金，平凡的泥土里培养出鲜活的生命。"]
    # wang = tokenizer(text)['input_ids'][1:-1]
    si = tokenizer.convert_tokens_to_ids(true_text)
    # [100, 4638, 100, 704, 100, 4708, 100, 4638, 100, 8024, 100, 4638, 100, 7027, 100, 1139, 100, 4638, 100, 511]
    # 这样UnKnown的词汇太多
    jie = tokenizer(new_text)['input_ids']
    # [[101, 2398, 1127, 4638, 3763, 2094, 704, 5943, 1419, 4708, 2140, 6586, 4638, 7942, 7032, 8024, 2398, 1127, 4638, 3799, 1759, 7027, 1824, 1075, 1139, 7831, 3833, 4638, 4495, 1462, 511, 102]]
    # 平，凡，的，沙，子
    # 我需要处理一下原数据集


    # print(wang)
    print(si)
    print(jie)
