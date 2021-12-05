import random
import math
from utils import loadDataAndFeature, dataSplit, PADDING, embd_name

LABELPAD = 0

label_map = {'padding': 0,
             'MajorClaim': 1,
             'Claim': 2,
             'Premise': 3,
             'Other': 4}

# 第一人称词表
first_person_list = ['I', 'me', 'my', 'mine', 'myself']

# 开头词表
forward_list = ['As a result', 'As the consequence', 'Because', 'Clearly', 'Consequently', 'Considering this subject',
                'Furthermore', 'Hence', 'leading to the consequence', 'so', 'So', 'taking account on this fact',
                'That is the reason why', 'The reason is that', 'Therefore', 'therefore', 'This means that',
                'This shows that', 'This will result', 'Thus', 'thus', 'Thus, it is clearly seen that',
                'Thus, it is seen', 'Thus, the example shows']

# 总结词表
backward_list = ['Additionally', 'As a matter of fact', 'because', 'Besides', 'due to', 'Finally', 'First of all',
                 'Firstly', 'for example', 'For example', 'For instance', 'for instance', 'Furthermore',
                 'has proved it', 'In addition', 'In addition to this', 'In the first place', 'is due to the fact that',
                 'It should also be noted', 'Moreover', 'On one hand', 'On the one hand', 'On the other hand',
                 'One of the main reasons', 'Secondly', 'Similarly', 'since', 'Since', 'So', 'The reason',
                 'To begin with', 'To offer an instance', 'What is more']

# 主题列表
thesis_list = ['All in all', 'All things considered', 'As far as I am concerned', 'Based on some reasons',
               'by analyzing both the views', 'considering both the previous fact', 'Finally',
               'For the reasons mentioned above', 'From explanation above', 'From this point of view', 'I agree that',
               'I agree with', 'I agree with the statement that', 'I believe', 'I believe that',
               'I do not agree with this statement', 'I firmly believe that', 'I highly advocate that',
               'I highly recommend', 'I strongly believe that', 'I think that', 'I think the view is',
               'I totally agree', 'I totally agree to this opinion', 'I would have to argue that',
               'I would reaffirm my position that', 'In conclusion', 'in conclusion', 'in my opinion', 'In my opinion',
               'In my personal point of view', 'in my point of view', 'In my point of view', 'In summary',
               'In the light of the facts outlined above', 'it can be said that', 'it is clear that',
               'it seems to me that', 'my deep conviction', 'My sentiments', 'Overall', 'Personally',
               'the above explanations and example shows that', 'This, however', 'To conclude', 'To my way of thinking',
               'To sum up', 'Ultimately']

# 反驳词表
rebuttal = ['Admittedly', 'although', 'Although', 'besides these advantages', 'but', 'But', 'Even though',
            'even though', 'However', 'Otherwise']

# 指示器
indicators = [forward_list, backward_list, thesis_list, rebuttal]


def labelEncode(labels):
    en_labels = []
    for labs in labels:
        en_labs = []
        for label in labs:
            en_labs.append(label_map[label])
        en_labels.append(en_labs)
    return en_labels

# Transformer中的encoder教程:https://zhuanlan.zhihu.com/p/366911389
# 可以直接encoder获得每个batch的输入
# input_ids_method1 = torch.tensor(tokenizer.encode(sentence, add_special_tokens=True))  # Batch size 1
# tensor([ 101, 7592, 1010, 2026, 2365, 2003, 3013, 2075, 1012,  102])

# tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
# is_word=False
def encodeBert(documents, labels, tokenizer, is_word=False):
    en_documents = []
    # 每行的（每个句子通过逗号划分开）
    for sentences in documents:
        length = len(sentences)
        out_sentences = []
        # 每一个句子
        for sentence in sentences:
            if is_word:
                seq = tokenizer.tokenize(''.join(sentence))
            else:
                # 先分词
                seq = tokenizer.tokenize(sentence)
                # ['hello', ',', 'my', 'son', 'is', 'cut', '##ing', '.']
            # 再转换为id，此时还没有转化为embedding
            seq = tokenizer.convert_tokens_to_ids(seq)
            # tensor([7592, 1010, 2026, 2365, 2003, 3013, 2075, 1012])
            # 并没有开头和结尾的标记：[cls]、[sep]
            out_sentences.append(seq)
        en_documents.append(out_sentences)

    en_labels = labelEncode(labels)

    return en_documents, en_labels

# title=True,is_word=False
def getEnglishSamplesBertId(in_file, tokenizer, title=False, is_word=False):

    # 返回句子列表，一行文本每句话的标签列表，每个行数据的每个句子的按顺序对应的六个特征：['gpos', 'lpos', 'ppos', 'gid', 'lid', 'pid']
    documents, labels, features = loadDataAndFeature(in_file, title)

    # 获取本文中每个句子的embedding组合；每个句子对应的label列表
    en_documents, en_labels = encodeBert(documents, labels, tokenizer, is_word)

    return en_documents, en_labels, features


# n_l = 40, is_cutoff=True
def sentencePaddingId(en_documents, labels, n_l, is_cutoff=True):
    pad_documents = []
    # 每行
    for sentences in en_documents:
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


# 输入：每个行数据的每个句子的按顺序对应的六个特征，获取本文中每个句子的embedding的id号，每个句子对应的label列表，Bert的tokenizer
# ???返回：(batch_n, doc_l, ft_size)
# 处理新增的手工特征
def featuresExtend(features, en_documents, en_labels, tokenizer):

    # str = "This is a tokenization example"
    # tokenized = tokenizer.tokenize(str)
    # ['this', 'Ġis', 'Ġa', 'Ġtoken', 'ization', 'Ġexample']
    # encoded = tokenizer.encode_plus(str)
    # encoded['input_ids']=[0, 42, 16, 10, 19233, 1938, 1246, 2]
    # decoded = tokenizer.decode(encoded['input_ids'])
    # '<s> this is a tokenization example</s>'

    # 在解码过程中将需要每个token映射到正确的input word

    # join()可以把字符串、元组、列表中的元素以指定的字符(分隔符)连接生成一个新的字符串
    # add_special_tokens 添加特殊标记，默认为True；如果设置为True，序列将使用相对于其模型的特殊标记进行编码
    # False: [1045, 2033, 2026, 3067, 2870]，相当于除去[cls][sep]
    # True: [101, 1045, 2033, 2026, 3067, 2870, 102]

    en_fp_list = tokenizer.encode(' '.join(first_person_list), add_special_tokens=False)

    # indicator：开头词列表，总结词列表，主题词列表，反驳列表
    # 进行encoder
    # 24，33，48，10
    en_indicators = []
    for indi in indicators:
        en_indi = []
        for m in indi:
            en_indi.append(tokenizer.encode(m, add_special_tokens=False))
        en_indicators.append(en_indi)

    # 本文通过合并指标特征和组件位置特征构建了一个特征向量：段落中的前面组件和后组件数量
    n_features = []
    # 对于每一篇文章
    for i, fts in enumerate(features):
        n_fts = []
        fd_c = 0
        # 减去padding和Others两个label之后的文章句子数
        bd_c = len(en_labels[i]) - en_labels[i].count(4) - en_labels[i].count(0)
        # 对于文章中的每一句话
        for j, ft in enumerate(fts):
            # ft[5]='pid'
            # first paragraph
            if ft[5] == 1:
                ft.append(1)
            else:
                ft.append(0)
            # ft[2]='ppos'
            # last paragraph    
            if ft[2] == 1:
                ft.append(1)
            else:
                ft.append(0)
            # first person indictor
            # 判断是否包含第一人称的id
            fp_flag = True
            for m in en_fp_list:
                if m in en_documents[i][j]:
                    ft.append(1)
                    fp_flag = False
                    break
            if fp_flag:
                ft.append(0)

            # indictors
            indi_list = [0, 0, 0, 0]
            for k, indi in enumerate(en_indicators):
                for m in indi:
                    # 先解码返回文字，判断当前句子中是否包含相应的文本
                    if tokenizer.decode(m) in tokenizer.decode(en_documents[i][j]):
                        indi_list[k] = 1
                        break
            # append函数直接将object整体当作一个元素追加到列表中，而extend函数则是将可迭代对象中的元素逐个追加到列表中
            ft.extend(indi_list)
            # number of preceding components  前面的组件数
            ft.append(fd_c)
            # number of following components  后面的组件数
            ft.append(bd_c)
            # 原本是not in，改成in
            if en_labels[i][j] in [0, 4]:
                fd_c += 1
                bd_c -= 1

            n_fts.append(ft)  # ft: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11] ，共15个数
        n_features.append(n_fts)
    return n_features


def batchGeneratorId(en_documents, labels, features, batch_n, is_random=False):
    data = list(zip(en_documents, labels, features))

    data.sort(key=lambda x: len(x[0]))
    for i in range(0, len(en_documents), batch_n):
        if is_random:
            random.seed()
            mid = random.randint(0, len(en_documents) - 1)
            # print(mid)
            start = max(0, mid - int(batch_n / 2))
            end = min(len(en_documents), mid + math.ceil(batch_n / 2))
        else:
            start = i
            end = i + batch_n

        b_data = data[start: end]

        b_docs, b_labs, b_ft = zip(*b_data)
        b_ft = list(b_ft)

        b_docs = list(b_docs)
        b_labs = list(b_labs)
        max_len = len(b_docs[-1])
        if len(b_docs[0]) == max_len:
            yield b_docs, b_labs, b_ft
        else:
            sen_len = len(b_docs[0][0])
            ft_len = len(b_ft[0][0])
            for j in range(len(b_docs)):
                if len(b_docs[j]) < max_len:
                    l = len(b_docs[j])
                    b_docs[j] = b_docs[j] + [PADDING * sen_len] * (max_len - l)
                    b_labs[j] = b_labs[j] + [LABELPAD] * (max_len - l)
                    b_ft[j] = b_ft[j] + [PADDING * ft_len] * (max_len - l)
                else:
                    break
            yield b_docs, b_labs, b_ft
