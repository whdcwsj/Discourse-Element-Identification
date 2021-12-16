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
# 每篇文章最多不超过99句话
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


# 每个句子的最大长度(最多四十个词)
def sentencePaddingId(documents, n_l=40, is_cutoff=True):
    pad_documents = []
    # 每个句子
    for sentence in documents:
        if len(sentence) % n_l:
            sentence = sentence + PADDING * (n_l - len(sentence) % n_l)
        if is_cutoff:
            pad_documents.append(sentence[0: n_l])

    return pad_documents


class BertDataset(Dataset):
    def __init__(self, config, data_path, add_title=True):
        super(BertDataset, self).__init__()
        self.config = config
        self.data_list = []
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_path)
        self.add_title = add_title

        # 返回每篇文章：句子列表(带titile)，每句话的标签列表，每个句子的按顺序对应的六个特征
        self.documents, labels, self.pos_features = loadDataAndFeature(data_path, title=self.add_title)

        # 所有文章，每片文章中所有句子的label，从文字转换为id
        self.id_labels = labelEncode(labels)


    def __getitem__(self, item):
        essay = self.documents[item]
        pos_item = self.pos_features[item]
        label_item = self.id_labels[item]

        # 将分散的中文单词，每句话合并到一起
        cn_essay = []
        # 每句话
        for i in range(len(essay)):
            # 每句话都放在一个列表中
            out_sentence = []
            temp_string = ''
            # 每句话中的单词
            for j in range(len(essay[i])):
                temp_string = temp_string + essay[i][j]
            out_sentence.append(temp_string)
            cn_essay.append(out_sentence)

        # 将子词列表转化为id的列表，无[CLS]和[SEP]
        token_id = []
        for i in range(len(cn_essay)):
            seq = self.tokenizer.tokenize(''.join(cn_essay[i]))
            sentence_id = self.tokenizer.convert_tokens_to_ids(seq)
            token_id.append(sentence_id)

        # token_id: [[7231, 6428], [100, 4761, 7231, 2218, 3121, 8024, 1587, 5811, 1920, 4183, 100, 1372,
        # pad_token_id: [[7231, 6428, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        pad_token_id = sentencePaddingId(token_id)

        # 格式转换
        pad_token_id = torch.tensor(pad_token_id, dtype=torch.float, device=self.config.device)
        pos_item = torch.tensor(pos_item, dtype=torch.float, device=self.config.device)[:, :6]
        label_item = torch.tensor(label_item, dtype=torch.long, device=self.config.device)


        return pad_token_id, pos_item, label_item

    def __len__(self):
        return len(self.documents)



if __name__ == '__main__':

    # config = Config(name='bert_classifier')
    # model = BertClassifier(config).to(config.device)
    # train_dataset = BertDataset(config, r'../data/train/train')
    # dataloader = DataLoader(train_dataset)
    # for data in dataloader:
    #     token_ids, masks, _, out, _ = data
    #     _output, _, _, _ = model(token_ids, masks)
    #     print()

    # documents, labels, pos_features = loadDataAndFeature('../data/new_Ch_dev.json', title=True)
    # print(len(pos_features[-1]))
    # print(len(documents[-1]))
    # kkk = pos_features[-1]
    # print(kkk[:, :6])

    # id_labels = labelEncode(labels)
    # print(len((features)))
    # print(documents[-3])

    # pad_documents, pad_labels = sentencePaddingId(documents, labels)
    # print(pad_documents[-3])
    # print(pad_labels[-3])

    # print(len(documents[-3]))
    # print(documents[-3])
    # print(labels[-3])
    # print(features[-3])


    # wang = documents[-1]
    # print(len(wang))
    # cn_essay = []
    #
    # # 每句话
    # for i in range(len(wang)):
    #     out_sentence = []
    #     temp_string = ''
    #     for j in range(len(wang[i])):
    #         temp_string = temp_string + wang[i][j]
    #     out_sentence.append(temp_string)
    #     cn_essay.append(out_sentence)
    #
    # print(cn_essay)

    # 英文的编码之后的样子，每个句子单独编码一次，包括title
    # [[2336, 2323, 5702, 2524, 2030, 2652, 4368, 1029, 2119, 2064, 5335, 2037, 2925],

    # tokenizer = BertTokenizer.from_pretrained(r'/home/wsj/bert_model/chinese/bert_chinese_L-12_H-768_A-12')
    # text = '今天天气真不错'
    # true_text = ["平凡", "的", "沙子", "中", "蕴含", "着", "宝贵", "的", "黄金", "，", "平凡", "的", "泥土", "里", "培养", "出", "鲜活", "的", "生命", "。"]
    # new_text = "平凡的沙子中蕴含着宝贵的黄金，平凡的泥土里培养出鲜活的生命。"
    # true_text1 = ["“", "知错", "就", "改", "，", "善莫大焉", "”", "只要", "我们", "能够", "重新", "改过", "，", "人格", "便", "会", "得到", "升华", "。"]
    # wang = tokenizer(text)['input_ids'][1:-1]
    # si = tokenizer.convert_tokens_to_ids(new_text)
    # si = tokenizer(new_text)
    # 该方法，中间会有空格
    # hhh = tokenizer.convert_tokens_to_string(true_text1)
    #
    #
    # # [100, 4638, 100, 704, 100, 4708, 100, 4638, 100, 8024, 100, 4638, 100, 7027, 100, 1139, 100, 4638, 100, 511]
    # # 这样UnKnown的词汇太多
    # # jie = tokenizer(new_text)['input_ids']
    # # [[101, 2398, 1127, 4638, 3763, 2094, 704, 5943, 1419, 4708, 2140, 6586, 4638, 7942, 7032, 8024, 2398, 1127, 4638, 3799, 1759, 7027, 1824, 1075, 1139, 7831, 3833, 4638, 4495, 1462, 511, 102]]
    # # 平，凡，的，沙，子

    # now = tokenizer(temp_string)['input_ids'][1:-1]
    # print(now)

    # print(wang)
    # print(si)
    # print(hhh)
    # print(jie)

    # {'input_ids': [[101, 7231, 6428, 102],
    #                [101, 100, 4761, 7231, 2218, 3121, 8024, 1587, 5811, 1920, 4183, 100, 1372, 6206, 2769, 812, 5543,
    #                 1916, 7028, 3173, 3121
    # qqq = tokenizer(cn_essay)
    # print(qqq)

    # eee = []
    # for i in range(len(cn_essay)):
    #     seq = tokenizer.tokenize(''.join(cn_essay[i]))
    #     ppp = tokenizer.convert_tokens_to_ids(seq)
    #     eee.append(ppp)
    # print(eee)
    # # print(len(eee))
    # nnn = sentencePaddingId(eee)
    # print(nnn)

    config = Config()
    dev_dataset = BertDataset(config=config, data_path=config.dev_data_path)
    dataloader = DataLoader(dev_dataset)
    i = 0
    for data in dataloader:
        token_ids, pos, label = data
        if i == 0:
            print(token_ids)
            print(pos)
            print(label)
        i += 1
