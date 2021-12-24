import utils
from model import *
from model_gru import *
from model_gate import *
from model_transformer import *

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import datetime
import os
import logging
import tqdm
from tqdm import *

from tensorboardX import SummaryWriter
import time
import argparse
from new_optimizer import ChildTuningOptimizer

import matplotlib.pyplot as plt

plt.switch_backend('Agg')

currenttime = time.localtime()

model_package_name = 'newbaseline_newstructure1_gate2_cat1'


# 固定随机数种子
def seed_torch(seed=1):
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)   # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)   # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# 输入：b_docs, b_labs, b_ft，p_embd = 'add'，device = 'cuda'
def list2tensor(x, y, ft, p_embd, device='cpu'):
    inputs = torch.tensor(x, dtype=torch.float, device=device)
    labels = torch.tensor(y, dtype=torch.long, device=device)

    tp = torch.tensor(ft, dtype=torch.float, device=device)[:, :, :6]
    # 句子长度补到40,word_embedding
    # (1,8,40,200)
    # (1,8)
    # (1,8,6)
    return inputs, labels, tp


# 输入：ft(一行文本中的句子对应的六个特征)
def getMask(ft, device='cpu'):
    slen = torch.tensor(ft, dtype=torch.long)[:, :, 6]

    s_n = slen.size(0) * slen.size(1)
    slen = slen.view(s_n)

    mask = torch.zeros((s_n, 40)) == 1
    for i, d in enumerate(slen):
        if d < 40:
            mask[i, d:] = 1
    if device == 'cuda':
        return mask.cuda()
    return mask


# model=STWithRSbySPP，X=按照max_len长度进行处理的句子的embedding，Y=每个句子对应的label列表，FT=每个行数据的每个句子的按顺序对应的六个特征
# title=True，is_mask=False
def train(model, X, Y, FT, is_gpu=False, epoch_n=10, lr=0.1, batch_n=100, title=False, is_mask=False):

    modelName = model.getModelName()
    if title:
        modelName += '_t'

    writer = SummaryWriter('./newlog/cn/' +  model_package_name + '/cn_' + modelName + '_' + time.strftime('%m-%d_%H.%M', currenttime))

    print(len(X))

    # 10%的数据作为验证集
    X_train, Y_train, ft_train, X_test, Y_test, ft_test = utils.dataSplit(X, Y, FT, 0.1)

    if (is_gpu):
        model.cuda()
        device = 'cuda'
    else:
        model.cpu()
        device = 'cpu'

    loss_function = nn.NLLLoss()

    optimizer = optim.SGD(model.parameters(), lr=lr)
    # optimizer = optim.Adam(model.parameters(), lr=3e-4)

    # optimizer = ChildTuningOptimizer.ChildTuningAdamW(model.parameters(), lr=lr)

    loss_list = []
    acc_list = []
    last_loss = 100
    c = 0
    best_epoch = -1

    last_acc, _, _ = test(model, X_test, Y_test, ft_test, device, title=title, is_mask=is_mask)

    for epoch in tqdm(range(epoch_n)):
        total_loss = 0
        gen = utils.batchGenerator(X_train, Y_train, ft_train, batch_n, is_random=True)
        i = 0
        model.train()
        for x, y, ft in gen:
            optimizer.zero_grad()  # 将梯度归零

            inputs, labels, tp = list2tensor(x, y, ft, model.p_embd, device)  # inputs:(50,34,40,200)

            if is_mask:
                mask = getMask(ft, device)
            else:
                mask = None

            if title:
                result = model(inputs, pos=tp, device=device, mask=mask)[:, 1:].contiguous()  # result: (batch_n, doc_l-1, class_n)
                labels = labels[:, 1:].contiguous()
            else:
                result = model(inputs, pos=tp, device=device, mask=mask)

            r_n = labels.size()[0] * labels.size()[1]
            result = result.contiguous().view(r_n, -1)
            labels = labels.view(r_n)

            loss = loss_function(result, labels)
            loss.backward()  # 反向传播计算得到每个参数的梯度
            optimizer.step()  # 通过梯度下降执行一步参数更新

            total_loss += loss.cpu().detach().numpy()
            i += 1

        aver_loss = total_loss / i
        loss_list.append(aver_loss)

        accuracy, dev_aver_loss, _ = test(model, X_test, Y_test, ft_test, device, title=title, is_mask=is_mask)
        acc_list.append(accuracy)

        writer.add_scalar("loss/train", aver_loss, epoch)
        writer.add_scalar("loss/dev", dev_aver_loss, epoch)
        writer.add_scalar("performance/accuracy", accuracy, epoch)

        if last_acc < accuracy:
            last_acc = accuracy
            if accuracy > 0.6:
                # 取每20个epoch中效果最好的
                torch.save(model, model_dir + '%s_%d_best.pk' % (modelName, int(epoch / 20) * 20))
            if epoch > 200:
                # 额外记录最好的那一个
                torch.save(model, model_dir + '%s_top.pk' % modelName)
                best_epoch = epoch

        # if (aver_loss > last_loss):
        #     c += 1
        #     if c == 10:
        #         lr = lr * 0.95
        #         optimizer.param_groups[0]['lr'] = lr
        #         c = 0
        # else:
        #     c = 0
        #     last_loss = aver_loss

        torch.save(model, model_dir + '%s_last.pk' % (modelName))

        # if (lr < 0.0001) or (aver_loss < 0.5):
        #     break

    # 若无最佳模型，跳过该步骤
    if best_epoch == -1:
        pass
    else:
        # top模型文件添加epoch记录
        oldname = model_dir + '%s_top.pk' % modelName
        newname = model_dir + '%s_epoch_%d_top.pk' % (modelName, best_epoch)
        os.rename(oldname, newname)

    writer.close()

    # plt.cla()
    # plt.plot(range(len(acc_list)), acc_list, range(len(loss_list)), loss_list)
    # plt.legend(['acc_list', 'loss_list'])
    # plt.savefig('./img/' + modelName + '.jpg')


# 训练集数据的10%，作为验证集
# X=按照max_len长度进行处理的句子的embedding，Y=每个句子对应的label列表，FT=每个行数据的每个句子的按顺序对应的六个特征
def test(model, X, Y, FT, device='cpu', batch_n=1, title=False, is_mask=False):
    loss_function = nn.NLLLoss()
    result_list = []
    label_list = []
    model.eval()
    total_loss = 0
    i = 0
    with torch.no_grad():
        # 返回：b_docs, b_labs, b_ft
        # 有一定排序的，先短后长
        gen = utils.batchGenerator(X, Y, FT, batch_n)
        for x, y, ft in gen:

            # (1,8,40,200)
            # (1,8)
            # (1,8,6)
            # tensor化
            inputs, labels, tp = list2tensor(x, y, ft, model.p_embd, device)

            if is_mask:
                mask = getMask(ft, device)
            else:
                mask = None

            if title:
                # (1,7,8)
                # [:, 1:],doc_l-1，减去title的长度
                # contiguous一般与transpose，permute，view搭配使用
                # 使用transpose或permute进行维度变换后，调用contiguous，然后方可使用view对维度进行变形

                # 如果不contiguous，直接view的话会报错
                # 1 transpose、permute等维度变换操作后，tensor在内存中不再是连续存储的
                # 而view操作要求tensor的内存连续存储，所以需要contiguous来返回一个contiguous copy；
                # 2 维度变换后的变量是之前变量的浅拷贝，指向同一区域，即view操作会连带原来的变量一同变形，这是不合法的，所以也会报错；
                # ---- 这个解释有部分道理，也即contiguous返回了tensor的深拷贝contiguous copy数据；

                result = model(inputs, pos=tp, device=device, mask=mask)[:, 1:].contiguous()  # result: (batch_n, doc_l, class_n)
                # (1,7)
                labels = labels[:, 1:].contiguous()  # labels:(batch_n, doc_l-(title))
            else:
                result = model(inputs, pos=tp, device=device, mask=mask)

            if labels.size()[0] == 0:
                print("labels的batch_n为0")
            if labels.size()[1] == 0:
                print("labels的doc_l-(title)为0")

            r_n = labels.size()[0] * labels.size()[1]
            # view：把原先tensor中的数据按照行优先的顺序排成一个一维的数据，然后按照参数组合成其他维度的tensor。

            # RuntimeError: cannot reshape tensor of 0 elements into shape [0, -1] because the unspecified dimension size -1 can be any value and is ambiguous
            # RuntimeError:无法将0元素的张量重塑为形状[0，-1]，因为未指定的维度大小-1可以是任何值，并且不明确
            result = result.contiguous().view(r_n, -1)  # result: (doc_l, class_n)  batch_n为1的情况下
            # label变成一维的
            labels = labels.view(r_n)

            loss = loss_function(result, labels)
            total_loss += loss.cpu().detach().numpy()
            i += 1

            result_list.append(result)
            label_list.append(labels)

    aver_loss = total_loss / i

    preds = torch.cat(result_list)  # preds:(2866,8)
    labels = torch.cat(label_list)  # labels:(2866,)
    t_c = 0
    # 混淆矩阵
    a = np.zeros((8, 8))
    l = preds.size()[0]  # 2866
    for i in range(l):
        p = preds[i].cpu().argmax().numpy()
        r = int(labels[i].cpu().numpy())
        a[r][p] += 1
        if p == r:
            t_c += 1
    accuracy = t_c / l

    # 7号是padding
    # print(a)
    # [[   0.    0.    0.    0.    0.  285.    0.    0.]
    #  [   0.    0.    0.    0.    0.   92.    0.    0.]
    #  [   0.    0.    0.    0.    0.  475.    0.    0.]
    #  [   0.    0.    0.    0.    0.  581.    0.    0.]
    #  [   0.    0.    0.    0.  104.  182.    0.    0.]
    #  [   0.    0.    0.    0.    3.   20.    0.    0.]
    #  [   0.    0.    0.    0.    0. 1124.    0.    0.]
    #  [   0.    0.    0.    0.    0.    0.    0.    0.]]

    return accuracy, aver_loss, a


def predict(model, x, ft, device='cpu', title=False):
    inputs, _, tp = list2tensor(x, [], ft, model.p_embd, device)

    if title:
        result = model(inputs, pos=tp, device=device)[:, 1:].contiguous()
    else:
        result = model(inputs, pos=tp, device=device)
    r_n = result.size()[0] * result.size()[1]
    result = result.contiguous().view(r_n, -1)
    return result.cpu().argmax(dim=1).tolist()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Chinese Discourse', usage='newtrain.py [<args>] [-h | --help]')
    parser.add_argument('--seed_num', default=1, type=int, help='Set seed num.')
    args = parser.parse_args()

    seed_torch(args.seed_num)

    in_file = './data/new_Ch_train.json'

    embed_filename = './embd/new_embeddings2.txt'
    title = True
    max_len = 40

    # 返回：获取本文中每个句子的embedding(单词组合)，每个句子对应的label列表，每个行数据的每个句子的按顺序对应的六个特征，vec_size
    en_documents, en_labels, features, vec_size = utils.getSamplesAndFeatures(in_file, embed_filename, title=title)

    # print(111)
    # print(np.array(en_documents[-1]).shape)
    # print(np.array(en_labels[-1]).shape)

    # 返回：按照max_len长度进行处理的句子的embedding，每个句子对应的label列表
    pad_documents, pad_labels = utils.sentence_padding(en_documents, en_labels, max_len, vec_size)

    is_mask = False

    # Introduction，Thesis，Main Idea，Evidence，Elaboration，Conclusion，Other，Padding
    class_n = 8
    batch_n = 50

    is_gpu = True
    if is_gpu and torch.cuda.is_available():
        pass
    else:
        is_gpu = False

    hidden_dim = 128
    sent_dim = 128

    p_embd = 'add'
    p_embd_dim = 16

    if p_embd in ['embd_b', 'embd_c', 'addv']:
        p_embd_dim = hidden_dim * 2

    if p_embd != 'embd_c':
        # 将后面三个'gid', 'lid', 'pid'根据前面三个'gpos'*40, 'lpos'*20, 'ppos'*10分别扩大
        # 返回大于等于该数值的最小整数
        features = utils.discretePos(features)

    # sent_dim用于设置每个句子的维度
    # tag_model = STWithRSbySPP_GRU_GATE(vec_size, hidden_dim, sent_dim, class_n, p_embd=p_embd, p_embd_dim=p_embd_dim,
    #                           pool_type='max_pool')

    # tag_model = STWithRSbySPP_NewStructure1(vec_size, hidden_dim, sent_dim, class_n, p_embd=p_embd, p_embd_dim=p_embd_dim,
    #                           pool_type='max_pool')

    tag_model = STWithRSbySPP_NewStructure1_Gate2(vec_size, hidden_dim, sent_dim, class_n, p_embd=p_embd, p_embd_dim=p_embd_dim,
                              pool_type='max_pool')

    if p_embd == 'embd_b':
        tag_model.posLayer.init_embedding()

    # 创建三个文件名
    if not os.path.exists('./newlog/cn/' + model_package_name):
        os.mkdir('./newlog/cn/' + model_package_name)
    if not os.path.exists('./newmodel/cn/' + model_package_name):
        os.mkdir('./newmodel/cn/' + model_package_name)
    if not os.path.exists('./newvalue/cn/' + model_package_name):
        os.mkdir('./newvalue/cn/' + model_package_name)

    model_dir = './newmodel/cn/' + model_package_name + '/' + tag_model.getModelName() + '-' + time.strftime('%m-%d_%H.%M', currenttime) + '_seed_' + str(args.seed_num) + '/'
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    print("start Chinese model training")
    starttime = datetime.datetime.now()
    train(tag_model, pad_documents, pad_labels, features, is_gpu, epoch_n=700, lr=0.2, batch_n=batch_n, title=title,
          is_mask=is_mask)
    endtime = datetime.datetime.now()
    print("本次seed为%d的训练耗时：" % int(args.seed_num))
    print(endtime - starttime)
