import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
import json
from src.config import Config
import random
import math


# 标签补足时，进行填充
LABELPAD = 0

# 句子长度不够时，进行填充
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


# 将中文的字词拼接起来，并进行编码
def chEncodeBert(documents, tokenizer):

    # 1、字词拼接
    cn_document = []

    # 每篇文章
    for essay in documents:
        essay_document = []
        # 每句话
        for i in range(len(essay)):
            # 每句话都放在一个列表中
            out_sentence = []
            temp_string = ''
            # 每句话中的单词
            for j in range(len(essay[i])):
                temp_string = temp_string + essay[i][j]
            out_sentence.append(temp_string)
            essay_document.append(out_sentence)

        cn_document.append(essay_document)

    # 2、将子词列表转化为id的列表，无[CLS]和[SEP]
    document_token_id = []
    # 每篇文章
    for i in range(len(cn_document)):
        essay_token_id = []
        # 每个句子
        for j in range(len(cn_document[i])):
            # 分词
            seq = tokenizer.tokenize(''.join(cn_document[i][j]))
            # 词转换为Bert ID
            sentence_id = tokenizer.convert_tokens_to_ids(seq)
            essay_token_id.append(sentence_id)
        document_token_id.append(essay_token_id)

    return cn_document, document_token_id


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


# 对于所有文章的句子进行处理
def essaySentencePaddingId(documents, n_l=40, is_cutoff=True):
    pad_documents = []
    # 每篇文章
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

    return pad_documents


# 一次输出一篇文章的Dataset
class BertSingleDataset(Dataset):
    def __init__(self, config, data_path):
        super(BertSingleDataset, self).__init__()
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_path)
        self.add_title = self.config.add_title

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


# 一次输出batch_size个文章的Dataset
class BertBatchDataset(Dataset):
    def __init__(self, config, data_path, batch_size=None, is_random=False, is_valid_test=False):
        super(BertBatchDataset, self).__init__()
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(self.config.bert_path)
        self.add_title = self.config.add_title
        # 验证集batch_size设置为1，训练集batch_size按照config来指定
        if is_valid_test:
            self.batch_size = batch_size
        else:
            self.batch_size = self.config.batch_size
        self.is_random = is_random

        # 返回每篇文章：句子列表(带titile)，每句话的标签列表，每个句子的按顺序对应的六个特征
        documents, labels, self.pos_features = loadDataAndFeature(data_path, title=self.add_title)
        # pos_features: [0.0625, 1.0, 0.2, 1, 1, 1]

        # 拼接中文字词，变成Bert的ID
        self.cn_document, self.document_token_id = chEncodeBert(documents=documents, tokenizer=self.tokenizer)

        # 填充，每个句子最多四十个词
        self.pad_document_id = essaySentencePaddingId(self.document_token_id)

        # 所有文章，每片文章中所有句子的label，从文字转换为id
        self.cn_labels = labelEncode(labels)  # [7, 1, 2, 3, 6, 2, 2, 3, 3, 2, 3, 2, 4, 4, 4, 4, 4]

    def __getitem__(self, item):

        # print("item:")
        # print(item)
        # print(type(item))

        data = list(zip(self.pad_document_id, self.cn_labels, self.pos_features))
        # 按文章句子个数从短到长排序
        data.sort(key=lambda x: len(x[0]))

        # 随机获取一个起始的ID
        start_id = item * self.batch_size
        if self.is_random:
            random.seed()
            # [a,b]
            mid = random.randint(0, len(self.pad_document_id) - 1)
            start = max(0, mid - int(self.batch_size / 2))
            # math.ceil()向上取整
            end = min(len(self.pad_document_id), mid + math.ceil(self.batch_size / 2))
        else:
            start = start_id
            end = start_id + self.batch_size

        batch_data = data[start: end]

        batch_document, batch_label, batch_feature = zip(*batch_data)
        # 输出的：<class 'tuple'>,需要列表化
        batch_feature = list(batch_feature)
        batch_document = list(batch_document)
        batch_label = list(batch_label)

        # 记录一个batch中文章包含的最长句子个数
        max_len_essay = len(batch_document[-1])

        # 如果当前批次句子长度均都一样
        if len(batch_document[0]) == max_len_essay:
            pass
        else:
            # 每个句子的单词数量
            sen_len = len(batch_document[0][0])
            # 特征的数量
            ft_len = len(batch_feature[0][0])
            # 遍历batch中的每篇文章
            for j in range(len(batch_document)):
                if len(batch_document[j]) < max_len_essay:
                    # 记录当前文章的句子数量
                    temp_l = len(batch_document[j])
                    batch_document[j] = batch_document[j] + [PADDING * sen_len] * (max_len_essay - temp_l)
                    batch_label[j] = batch_label[j] + [LABELPAD] * (max_len_essay - temp_l)
                    batch_feature[j] = batch_feature[j] + [PADDING * ft_len] * (max_len_essay - temp_l)
                else:
                    # 长度相同，则可以跳出本层for循环
                    break

        # 格式转换
        pad_token_id = torch.tensor(batch_document, dtype=torch.int, device=self.config.device)
        pos_item = torch.tensor(batch_feature, dtype=torch.float, device=self.config.device)[:, :, :6]
        label_item = torch.tensor(batch_label, dtype=torch.long, device=self.config.device)

        return pad_token_id, pos_item, label_item

    def __len__(self):
        return int(math.ceil(len(self.pad_document_id)/self.batch_size))



# 正规的Bert数据加载装置，自带attention_mask
class BertFormalBatchDataset(Dataset):
    def __init__(self, config, data_path, batch_size=None, is_random=False, is_valid_test=False):
        super(BertFormalBatchDataset, self).__init__()
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(self.config.bert_path)
        self.add_title = self.config.add_title
        # 验证集batch_size设置为1，训练集batch_size按照config来指定
        if is_valid_test:
            self.batch_size = batch_size
        else:
            self.batch_size = self.config.batch_size
        self.is_random = is_random

        # 返回每篇文章：句子列表(带titile)，每句话的标签列表，每个句子的按顺序对应的六个特征
        documents, labels, self.pos_features = loadDataAndFeature(data_path, title=self.add_title)
        # pos_features: [0.0625, 1.0, 0.2, 1, 1, 1]

        # 拼接中文字词，变成Bert的ID
        self.cn_document, self.document_token_id = chEncodeBert(documents=documents, tokenizer=self.tokenizer)

        # 填充，每个句子最多四十个词
        self.pad_document_id = essaySentencePaddingId(self.document_token_id)

        # 所有文章，每片文章中所有句子的label，从文字转换为id
        self.cn_labels = labelEncode(labels)  # [7, 1, 2, 3, 6, 2, 2, 3, 3, 2, 3, 2, 4, 4, 4, 4, 4]

    def __getitem__(self, item):

        # print("item:")
        # print(item)
        # print(type(item))

        data = list(zip(self.pad_document_id, self.cn_labels, self.pos_features))
        # 按文章句子个数从短到长排序
        data.sort(key=lambda x: len(x[0]))

        # 随机获取一个起始的ID
        start_id = item * self.batch_size
        if self.is_random:
            random.seed()
            # [a,b]
            mid = random.randint(0, len(self.pad_document_id) - 1)
            start = max(0, mid - int(self.batch_size / 2))
            # math.ceil()向上取整
            end = min(len(self.pad_document_id), mid + math.ceil(self.batch_size / 2))
        else:
            start = start_id
            end = start_id + self.batch_size

        batch_data = data[start: end]

        batch_document, batch_label, batch_feature = zip(*batch_data)
        # 输出的：<class 'tuple'>,需要列表化
        batch_feature = list(batch_feature)
        batch_document = list(batch_document)
        batch_label = list(batch_label)

        # 记录一个batch中文章包含的最长句子个数
        max_len_essay = len(batch_document[-1])

        # 如果当前批次句子长度均都一样
        if len(batch_document[0]) == max_len_essay:
            pass
        else:
            # 每个句子的单词数量
            sen_len = len(batch_document[0][0])
            # 特征的数量
            ft_len = len(batch_feature[0][0])
            # 遍历batch中的每篇文章
            for j in range(len(batch_document)):
                if len(batch_document[j]) < max_len_essay:
                    # 记录当前文章的句子数量
                    temp_l = len(batch_document[j])
                    batch_document[j] = batch_document[j] + [PADDING * sen_len] * (max_len_essay - temp_l)
                    batch_label[j] = batch_label[j] + [LABELPAD] * (max_len_essay - temp_l)
                    batch_feature[j] = batch_feature[j] + [PADDING * ft_len] * (max_len_essay - temp_l)
                else:
                    # 长度相同，则可以跳出本层for循环
                    break

        # 格式转换
        pad_token_id = torch.tensor(batch_document, dtype=torch.int, device=self.config.device)
        pos_item = torch.tensor(batch_feature, dtype=torch.float, device=self.config.device)[:, :, :6]
        label_item = torch.tensor(batch_label, dtype=torch.long, device=self.config.device)

        return pad_token_id, pos_item, label_item

    def __len__(self):
        return int(math.ceil(len(self.pad_document_id)/self.batch_size))


if __name__ == '__main__':

    # tokenizer = BertTokenizer.from_pretrained(r'/home/wsj/bert_model/chinese/bert_chinese_L-12_H-768_A-12')
    # documents, labels, pos_features = loadDataAndFeature('../data/new_Ch_train.json', title=True)
    #
    # print("pos_feature:")
    # print(len(pos_features[-1]))
    # print(pos_features[-1])
    # print(len(documents[-1]))
    # kkk = pos_features[-1]
    # kk = np.array(kkk)
    # print(kk[:, :6])

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

    # cn_document, document_token_id = chEncodeBert(documents, tokenizer)
    #
    # print("cn_document:")
    # print(len(cn_document[-1]))
    # print(cn_document[-1])
    #
    # print("document_token_id:")
    # print(len(document_token_id[-1]))
    # print(document_token_id[-1])
    #
    # cn_labels = labelEncode(labels)
    # print("cn_labels:")
    # print(len(cn_labels[-1]))
    # print(cn_labels[-1])
    #
    # pad_document_id = essaySentencePaddingId(document_token_id)
    #
    # print("pad_document_id:")
    # print(len(pad_document_id[-1]))
    # print(pad_document_id[-1])
    #
    # batch_n = 50
    # is_random = True
    #
    # data = list(zip(pad_document_id, cn_labels, pos_features))
    # print("data before:")
    # print(len(data[-1][0]))
    # print(data[-1][0])
    #
    # data.sort(key=lambda x: len(x[0]))
    # print("data after:")
    # print(len(data[-1][0]))
    # print(data[-1][0])
    # print("*****************************************")
    #
    # for i in range(0, len(pad_document_id), batch_n):
    #     if is_random:
    #         random.seed()
    #         # [a,b]
    #         mid = random.randint(0, len(pad_document_id) - 1)
    #         # print(mid)
    #         start = max(0, mid - int(batch_n / 2))
    #         # math.ceil()向上取整
    #         end = min(len(pad_document_id), mid + math.ceil(batch_n / 2))
    #     else:
    #         start = i
    #         end = i + batch_n
    #
    #     b_data = data[start: end]
    #
    #     b_docs, b_labs, b_ft = zip(*b_data)
    #     b_ft = list(b_ft)
    #
    #     b_docs = list(b_docs)
    #     b_labs = list(b_labs)
    #     max_len = len(b_docs[-1])
    #
    #     if i == 0:
    #         print("batch中的数据：")
    #         print("b_docs:")
    #         print(len(b_docs[0]))
    #         print(b_docs[0])
    #         print("b_ft：")
    #         print(len(b_ft[0]))
    #         print(b_ft[0])
    #         print("b_labs:")
    #         print(len(b_labs[0]))
    #         print(b_labs[0])
    #         print("当前批次最大文章长度:")
    #         print(max_len)
    #
    #         # 如果当前批次句子长度均都一样
    #         if len(b_docs[0]) == max_len:
    #             print(b_docs)
    #             print(b_labs)
    #             print(b_ft)
    #         else:
    #             # 每个句子的单词数量
    #             sen_len = len(b_docs[0][0])
    #             # 特征的数量
    #             ft_len = len(b_ft[0][0])
    #             for j in range(len(b_docs)):
    #                 if len(b_docs[j]) < max_len:
    #                     l = len(b_docs[j])
    #                     b_docs[j] = b_docs[j] + [PADDING * sen_len] * (max_len - l)
    #                     b_labs[j] = b_labs[j] + [LABELPAD] * (max_len - l)
    #                     b_ft[j] = b_ft[j] + [PADDING * ft_len] * (max_len - l)
    #                 else:
    #                     break
    #
    #             print("文章长度不一样之后的修改：")
    #             print(len(b_docs[0]))
    #             print(len(b_labs[0]))
    #             print(len(b_ft[0]))
    #             print(max_len)



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



    # 1、测试BertSingleDataset
    # config = Config(name='wsj_bert_test')
    # dev_dataset = BertSingleDataset(config=config, data_path=config.dev_data_path)
    # dataloader = DataLoader(dev_dataset, batch_size=1)
    # i = 0
    # for data in dataloader:
    #     if i == 0:
    #         token_ids, pos, label = data
    #
    #         print(token_ids.shape)  # torch.Size([1, 30, 40])
    #         print(pos.shape)  # torch.Size([1, 30, 6])
    #         print(label.shape)  # torch.Size([1, 30])
    #
    #     i += 1



    # # 2、测试BertBatchDataset
    # config = Config(name='wsj_bert_test')
    # # dev_dataset = BertBatchDataset(config=config, data_path=config.test_data_path, is_random=True)
    # dev_dataset = BertBatchDataset(config=config, data_path='../data/Ch_test.json', batch_size=1, is_valid_test=True)
    # # dev_dataset = BertBatchDataset(config=config, data_path=config.dev_data_path, batch_size=1, is_valid_test=True)
    #
    # dataloader = DataLoader(dev_dataset, batch_size=1)
    # i = 0
    # token_ids = None
    # pos = None
    # label = None
    # for data in dataloader:
    #     if i == 0:
    #         token_ids, pos, label = data
    #
    #         # print(token_ids.shape)  # torch.Size([1, 30, 25, 40])
    #         # print(pos.shape)  # torch.Size([1, 30, 25, 6])
    #         # print(label.shape)  # torch.Size([1, 30, 25])
    #     i += 1
    #
    # print("原始的shape:")
    # print(token_ids.shape)
    # print(pos.shape)
    # print(label.shape)
    #
    # new_token_id = token_ids.squeeze(0)
    # print(new_token_id.shape)
    #
    # print("------------------")
    # # print(new_token_id[0])
    #
    # # bert的输出：last_hidden_state, pooler_output, all_hidden_states, all_attentions
    # # 1、last_hidden_state：shape是(batch_size, sequence_length, hidden_size)，hidden_size=768，它是模型最后一层输出的隐藏状态
    # # 2、pooler_output：shape是(batch_size, hidden_size)，这是序列的第一个token(classification token)的最后一层的隐藏状态；代表该句句子向量【CLS】
    # # 3、hidden_states：输出可选项，如果输出，需要指定config.output_hidden_states=True
    # # 它也是一个元组，它的第一个元素是embedding，其余元素是各层的输出，每个元素的形状是(batch_size, sequence_length, hidden_size)
    # # 4、attentions：输出可选项，如果输出，需要指定config.output_attentions=True
    # # 它也是一个元组，它的元素是每一层的注意力权重，用于计算self-attention heads的加权平均值
    #
    # bert_temp = BertModel.from_pretrained(config.bert_path).to(config.device)
    #
    # temp_batch_output = []
    # for i in range(new_token_id.shape[0]):
    #     embedding = bert_temp(new_token_id[i])
    #     last_hidden_state = embedding[0]
    #     last_hidden_state = last_hidden_state.to(config.device)
    #     temp_batch_output.append(last_hidden_state)
    #
    # batch_bert_output = torch.stack(temp_batch_output, dim=0)
    # print(batch_bert_output.shape)
    # print(111)

    tokenizer = BertTokenizer.from_pretrained(r'/home/wsj/bert_model/chinese/chinese_bert_wwm_pytorch')

    # 返回每篇文章：句子列表(带titile)，每句话的标签列表，每个句子的按顺序对应的六个特征
    document, labels, pos_features = loadDataAndFeature('../data/Ch_test.json', title=True)

    documents = document[-2:]

    # temp_document = document[-1]
    # documents = []
    # documents.append(temp_document)

    # 1、字词拼接
    cn_document = []

    # 每篇文章
    for essay in documents:
        essay_document = []
        # 每句话
        for i in range(len(essay)):
            # 每句话都放在一个列表中
            out_sentence = []
            temp_string = ''
            # 每句话中的单词
            for j in range(len(essay[i])):
                temp_string = temp_string + essay[i][j]
            out_sentence.append(temp_string)
            essay_document.append(out_sentence)

        cn_document.append(essay_document)

    # 2、将子词列表转化为id的列表，带[CLS]和[SEP]
    document_token_id = []
    document_mask = []
    # 每篇文章
    for i in range(len(cn_document)):
        essay_token_id = []
        essay_attention_mask = []
        # 每个句子
        for j in range(len(cn_document[i])):
            # max_length：序列的最大长度
            # add_special_tokens：是否在序列前添加 [CLS]，结尾添加 [SEP]
            # truncation_strategy：当传入 max_length 时，这个参数才生效，表示截断策略
            # padding：是否填充
            # return_tensors：返回的张量类型(两种：'tf'/'pt')
            # tokenized = tokenizer(cn_document[i][j], max_length=100, add_special_tokens=True, truncation=True, padding=True, return_tensors="pt")
            tokenized = tokenizer(cn_document[i][j], return_tensors="pt")
            token_ids = tokenized['input_ids']
            masks = tokenized['attention_mask']

            essay_token_id.append(token_ids)
            essay_attention_mask.append(masks)

        document_token_id.append(essay_token_id)
        document_mask.append(essay_attention_mask)

    print(111)