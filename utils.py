import json
import os, sys
import numpy as np
import math
from collections import Counter

import random
# random.seed(312)

UNKNOWN = [0]
PADDING = [0]
LABELPAD = 7

embd_name = ['embd', 'embd_a', 'embd_b', 'embd_c']

# 标签名称对应的ID号
label_map = {'Introduction': 0,
             'Thesis': 1,
             'Main Idea': 2,
             'Evidence': 3,
             'Conclusion': 4,
             'Other': 5,
             'Elaboration': 6,
             'padding': 7}

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

# 输入：数据集，title=True
def loadDataAndFeature(in_file, title=False, max_len=99):
    labels = []
    documents = []
    ft_list = ['gpos', 'lpos', 'ppos', 'gid', 'lid', 'pid']
    features = []
    # 中文数据集每一行：file,title,score,sents,label,gid,lid,pid
    with open(in_file, 'r', encoding='utf8') as f:
        for line in f:
            ft = []
            # 解码json数据，返回{'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 'wsj'}
            load_dict = json.loads(line)
            # 返回新的参数字典，获取相对位置['gpos', 'lpos', 'ppos']
            load_dict = getRelativePos(load_dict)
            # 数据中是否添加title
            if title:
                if ('slen' in load_dict) and ('slen' not in ft_list):
                    ft_list.append('slen')

                load_dict['sents'].insert(0, load_dict['title'])
                load_dict['labels'].insert(0, 'padding')
                for k in ft_list:
                    load_dict[k].insert(0, 0)

                # 记录title的长度
                if 'slen' in load_dict:
                    load_dict['slen'][0] = len(load_dict['title'])

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

# 输入：标签列表（此时外面有两层[]）
def labelEncode(labels):
    en_labels = []
    for labs in labels:
        en_labs = []
        for label in labs:
            # if not label in label_map:
                # print(label)
                # continue
            en_labs.append(label_map[label])
        en_labels.append(en_labs)
    return en_labels

# 句子列表，标签列表，embedding，中文vector_size：200
def encode(documents, labels, embed_map, vec_size):

    en_documents = []
    # 每行的（每个句子通过列表分开）
    for sentences in documents:
        length = len(sentences)
        out_sentences = []
        # 每个句子
        for sentence in sentences:
            seq = [embed_map[w] if w in embed_map else UNKNOWN * vec_size for w in sentence]
            out_sentences.append(seq)
        en_documents.append(out_sentences)

    en_labels = labelEncode(labels)

    return en_documents, en_labels


# 输入：获取本文中每个句子的embedding(单词组合)，每个句子对应的label列表，句子最大长度max_len=40，中文vec_size=200
# is_cutoff=True
def sentence_padding(en_documents, labels, n_l, vec_size, is_cutoff=True, dgl=False):
    pad_documents = []
    # 每行的
    for sentences in en_documents:
        length = len(sentences)
        out_sentences = []
        # 每个句子
        for sentence in sentences:
            # 长度不等于0/n_l的情况
            if len(sentence) % n_l:
                # 每max_len个长度，剩余空位置补PADDING[0]
                sentence = sentence + [PADDING * vec_size] * (n_l - len(sentence) % n_l)
            # 截断保留前max_len个位置
            if is_cutoff:
                out_sentences.append(sentence[0: n_l])
            # 不截断的话，每max_len个长度都加入句子
            else:
                for i in range(0, len(sentence), n_l):
                    out_sentences.append(sentence[i: i+n_l])
                    # 还需要label填充
        pad_documents.append(out_sentences)
    pad_labels = labels
    return pad_documents, pad_labels


# 每个句子的最大长度n_l:40，不足的补[0]
# is_cutoff = True，截断超过长度40的
def sentence_padding_dgl(en_documents, labels, n_l, vec_size, is_cutoff=True):
    pad_documents = []
    # 记录每篇文章的实际句子长度
    out_essay_length = []
    # 每行的/每篇文章的(n个句子)
    for sentences in en_documents:
        # 每篇文章的实际句子个数
        length = len(sentences)
        out_essay_length.append(int(length))
        out_sentences = []
        # 其中的每一个句子
        for sentence in sentences:
            # 长度不等于0/n_l的情况
            # len(sentence) 相当于每个句子中的单词数
            if len(sentence) % n_l:
                # 每max_len个长度，剩余空位置补PADDING[0]
                sentence = sentence + [PADDING * vec_size] * (n_l - len(sentence) % n_l)
            # 截断保留前max_len个位置
            if is_cutoff:
                out_sentences.append(sentence[0: n_l])
            # 不截断的话，每max_len个长度都加入句子
            else:
                for i in range(0, len(sentence), n_l):
                    out_sentences.append(sentence[i: i+n_l])
                    # 还需要label填充
        pad_documents.append(out_sentences)
    pad_labels = labels
    return pad_documents, pad_labels, out_essay_length


def loadEmbeddings(embed_filename):
    embed_map = {}
    with open(embed_filename, 'r', encoding='utf-8') as f:
        lenth = f.readline()
        for line in f:
            # 去除这行文本的最后一个字符(换行符)后剩下的部分
            line = line[:-1]
            embed = line.split(' ')
            # 计算向量长度，跳过第一个字符
            vec_size = len(embed[1:])
            # 放到字典里面，vector数值float化
            embed_map[embed[0]] = [float(x) for x in embed[1:]]
    # embed_map['，'] = embed_map[',']
    return embed_map, vec_size


def featuresExtend(features, documents):
    pass

# 输入：数据集，embedding文件，title = True
# 返回：获取本文中每个句子的embedding组合(单词组合)，每个句子对应的label列表，每个行数据的每个句子的按顺序对应的六个特征，vec_size
def getSamplesAndFeatures(in_file, embed_filename, title=False, extend_f=False):

    print('load Embeddings...')

    # 返回embedding对应的字典, vector的长度(中文的embedding为200)
    embed_map, vec_size = loadEmbeddings(embed_filename)

    # 返回句子列表，一行文本每句话的标签列表，每个行数据的每个句子的按顺序对应的六个特征：['gpos', 'lpos', 'ppos', 'gid', 'lid', 'pid']
    documents, labels, features = loadDataAndFeature(in_file, title)

    # 进行编码
    # 获取本文中每个句子的embedding组合(单词组合)；每个句子对应的label列表
    en_documents, en_labels = encode(documents, labels, embed_map, vec_size)

    # 这个没用上
    if extend_f:
        features = featuresExtend(features, documents)
    # pad_documents, pad_labels = sentence_padding(en_documents, en_labels, 30, vec_size)

    return en_documents, en_labels, features, vec_size





# X=按照max_len长度进行处理的句子的embedding，Y=每个句子对应的label列表，FT=每个行数据的每个句子的按顺序对应的六个特征，batch_n = 1
# train的时候，is_random=True，batch_n = 50
def batchGenerator(en_documents, labels, features, batch_n, is_random=False):
    # multilabel = 0
    if type(labels[0][0]) in (int, str):
        mutiLabel = 0
    else:
        mutiLabel = len(labels[0])
    # 107条验证集，打包为元组的列表
    data = list(zip(en_documents, labels, features))

    # 根据zip中的第一个元素en_documents排序
    data.sort(key=lambda x: len(x[0]))
    # 有多少篇essay
    for i in range(0, len(en_documents), batch_n):
        if is_random:
            random.seed()
            mid = random.randint(0,len(en_documents)-1)
            # print(mid)
            start = max(0, mid - int(batch_n/2))
            end = min(len(en_documents), mid + math.ceil(batch_n/2))
        else:
            start = i
            end = i + batch_n
        # print(start, end)
        b_data = data[start: end]
        # b_data = data[i: i+batch_n]

        # *b_data相当于解压，返回二维矩阵形式
        b_docs, b_labs, b_ft = zip(*b_data)
        b_ft = list(b_ft)
        b_docs = list(b_docs)
        b_labs = list(b_labs)
        # 所有文章中的句子的最大个数，因为上面已经进行了sort()，从小到大排序
        max_len = len(b_docs[-1])
        # batch_n=1的情况，直接yield输出
        # yield每次返回结果之后函数并没有退出，而是每次遇到yield关键字后返回相应结果，并保留函数当前的运行状态，等待下一次的调用
        if len(b_docs[0]) == max_len:
            yield b_docs, b_labs, b_ft
        else:
            sen_len = len(b_docs[0][0])    # sen_len:40(句子最大长度)
            vec_size = len(b_docs[0][0][0])    # vec_size:200
            ft_len = len(b_ft[0][0])   # ft_len:6

            for j in range(len(b_docs)):
                if len(b_docs[j]) < max_len:
                    l = len(b_docs[j])
                    # 每个batch不足最大句子个数的位置填充PADDING
                    b_docs[j] = b_docs[j] + [[PADDING * vec_size] * sen_len] * (max_len - l)
                    # multilabel = 0
                    if not mutiLabel:
                        # 给添加的padding句子打label：LABELPAD → 7
                        b_labs[j] = b_labs[j] + [LABELPAD] * (max_len - l)
                    else:
                        b_labs[j] = [b_labs[j][0] + ([LABELPAD]) * (max_len - l),
                                     b_labs[j][1] + PADDING * (max_len - l)]

                    # 给添加的padding句子打feature：PADDING
                    b_ft[j] = b_ft[j] + [PADDING * ft_len] * (max_len - l)
                else:
                    break
            yield b_docs, b_labs, b_ft


# 在这里进行每个文章句子数目的统一(只有在train深处才进行该操作)
# X=按照max_len长度进行处理的句子的embedding，Y=每个句子对应的label列表，FT=每个行数据的每个句子的按顺序对应的六个特征，每篇文章的实际句子数量
# test的时候，batch_n = 1
# train的时候，is_random=True，batch_n = 50
def batchGenerator_dgl(en_documents, labels, features, essay_length, batch_n, is_random=False):
    # multilabel = 0
    if type(labels[0][0]) in (int, str):
        mutiLabel = 0
    else:
        mutiLabel = len(labels[0])
    # 107条验证集，打包为元组的列表
    data = list(zip(en_documents, labels, features, essay_length))

    # 根据zip中的第一个元素en_documents排序
    data.sort(key=lambda x: len(x[0]))
    # 有多少篇essay
    for i in range(0, len(en_documents), batch_n):
        if is_random:
            random.seed()
            mid = random.randint(0,len(en_documents)-1)
            # print(mid)
            start = max(0, mid - int(batch_n/2))
            end = min(len(en_documents), mid + math.ceil(batch_n/2))
        else:
            start = i
            end = i + batch_n
        # print(start, end)
        b_data = data[start: end]
        # b_data = data[i: i+batch_n]

        # *b_data相当于解压，返回二维矩阵形式
        b_docs, b_labs, b_ft, b_essay_length = zip(*b_data)
        b_ft = list(b_ft)
        b_docs = list(b_docs)
        b_labs = list(b_labs)   #  [[7, 0, 0, 2, 2, 2, 2, 4],[……],……]
        b_essay_length = list(b_essay_length)  # [8,7,……]
        # 所有文章中的句子的最大个数，因为上面已经进行了sort()，从小到大排序
        max_len = len(b_docs[-1])
        # batch_n=1的情况，直接yield输出
        # yield每次返回结果之后函数并没有退出，而是每次遇到yield关键字后返回相应结果，并保留函数当前的运行状态，等待下一次的调用
        if len(b_docs[0]) == max_len:
            yield b_docs, b_labs, b_ft, b_essay_length
        else:
            sen_len = len(b_docs[0][0])    # sen_len:40(句子最大长度)
            vec_size = len(b_docs[0][0][0])    # vec_size:200
            ft_len = len(b_ft[0][0])   # ft_len:6

            # len(b_docs)：当前batch的文章数目
            for j in range(len(b_docs)):
                if len(b_docs[j]) < max_len:
                    l = len(b_docs[j])
                    # 每个batch不足最大句子个数的位置填充PADDING
                    b_docs[j] = b_docs[j] + [[PADDING * vec_size] * sen_len] * (max_len - l)
                    # multilabel = 0
                    if not mutiLabel:
                        # 给添加的padding句子打label：LABELPAD → 7(Others标签)
                        b_labs[j] = b_labs[j] + [LABELPAD] * (max_len - l)
                    else:
                        b_labs[j] = [b_labs[j][0] + ([LABELPAD]) * (max_len - l),
                                     b_labs[j][1] + PADDING * (max_len - l)]

                    # 给添加的padding句子打feature：PADDING
                    b_ft[j] = b_ft[j] + [PADDING * ft_len] * (max_len - l)
                else:
                    break
            yield b_docs, b_labs, b_ft, b_essay_length


# pad_documents, pad_labels
# 输入：X=按照max_len长度进行处理的句子的embedding，Y=每个句子对应的label列表，FT=每个行数据的每个句子的按顺序对应的六个特征
def dataSplit(X, Y, ft=None, p=0.1):
    # 使用random.seed()，而不是np.numpy.seed()，抱着测试集的不变性，避免主项目中的随机数变化干扰数据划分
    random.seed(312)
    # 返回范围[a,b]内的随机整数
    # 调训10%的数字作为验证集，也就是训练中的测试集
    test_idx = [random.randint(0,len(X)-1) for _ in range(int(len(X)*p))]
    X_test = []
    Y_test = []
    ft_test = []
    X_train = []
    Y_train = []
    ft_train = []
    for i in range(len(X)):
        if i in test_idx:
            X_test.append(X[i])
            Y_test.append(Y[i])
            if ft:
                ft_test.append(ft[i])
        else:
            X_train.append(X[i])
            Y_train.append(Y[i])
            if ft:
                ft_train.append(ft[i])
    if not ft:
        ft_test = None
        ft_train = None

    return X_train, Y_train, ft_train, X_test, Y_test, ft_test


# 输入：X=按照max_len长度进行处理的句子的embedding，Y=每个句子对应的label列表，FT=每个行数据的每个句子的按顺序对应的六个特征，essay_len每篇文章中的句子长度
def dataSplit_dgl(X, Y, ft=None, essay_len=None, p=0.1):
    # 使用random.seed()，而不是np.numpy.seed()，抱着测试集的不变性，避免主项目中的随机数变化干扰数据划分
    random.seed(312)
    # 返回范围[a,b]内的随机整数
    # 调训10%的数字作为验证集，也就是训练中的测试集
    test_idx = [random.randint(0,len(X)-1) for _ in range(int(len(X)*p))]
    X_test = []
    Y_test = []
    ft_test = []
    test_essay_len = []
    X_train = []
    Y_train = []
    ft_train = []
    train_essay_len = []
    for i in range(len(X)):
        if i in test_idx:
            X_test.append(X[i])
            Y_test.append(Y[i])
            if ft:
                ft_test.append(ft[i])
            if essay_len:
                test_essay_len.append(int(essay_len[i]))
        else:
            X_train.append(X[i])
            Y_train.append(Y[i])
            if ft:
                ft_train.append(ft[i])
            if essay_len:
                train_essay_len.append(int(essay_len[i]))
    if not ft:
        ft_test = None
        ft_train = None
    if not essay_len:
        test_essay_len = None
        train_essay_len = None

    return X_train, Y_train, ft_train, train_essay_len, X_test, Y_test, ft_test, test_essay_len


# 输入：每个行数据的每个句子的按顺序对应的六个特征:['gpos', 'lpos', 'ppos', 'gid', 'lid', 'pid']
# 将'gid', 'lid', 'pid'根据'gpos', 'lpos', 'ppos'分别扩大
def discretePos(features):
    for feat in features:
        for f in feat:
            # math.ceil() 返回大于或等于一个给定数字的最小整数。
            f[3] = math.ceil(f[0]*40)
            f[4] = math.ceil(f[1]*20)
            f[5] = math.ceil(f[2]*10)
    return features
