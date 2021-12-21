import utils
from model import *
from model_gru import *
from model_dgl import *

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


# model_package_name = 'dgl0.6_pos1_dgl3_base_tuning'
# dgl0.6_pos1_dgl3_base_tuning_adam_without0.5


# 固定随机数种子
def seed_torch(seed=1):
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# 输入：b_docs, b_labs, b_ft，p_embd = 'add'，device = 'cuda'
def list2tensor_dgl(x, y, ft, e_len, p_embd, device='cpu'):
    inputs = torch.tensor(x, dtype=torch.float, device=device)
    labels = torch.tensor(y, dtype=torch.long, device=device)

    tp = torch.tensor(ft, dtype=torch.float, device=device)[:, :, :6]
    e_len = torch.tensor(e_len, dtype=torch.long, device=device)

    return inputs, labels, tp, e_len


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


# tag_model, pad_documents, pad_labels, features
# model，X=按照max_len长度进行处理的句子的embedding，Y=每个句子对应的label列表，FT=每个行数据的每个句子的按顺序对应的六个特征
# title=True，is_mask=False
def train(model, X, Y, FT, essay_len, is_gpu=False, epoch_n=10, lr=0.1, batch_n=100, title=False, is_mask=False):
    modelName = model.getModelName()
    if title:
        modelName += '_t'

    writer = SummaryWriter(
        './newlog/cn/dgl/' + model_package_name + '/cn_' + modelName + '_' + time.strftime('%m-%d_%H.%M', currenttime))

    # 10%的数据作为验证集
    X_train, Y_train, ft_train, essay_len_train, X_test, Y_test, ft_test, essay_len_test = utils.dataSplit_dgl(X, Y, FT,
                                                                                                               essay_len,
                                                                                                               0.1)

    if (is_gpu):
        model.cuda()
        device = 'cuda'
    else:
        model.cpu()
        device = 'cpu'

    loss_function = nn.NLLLoss()

    # optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # adam最优学习率为3e-4
    # 另一种建议1e-5

    optimizer = optim.SGD(model.parameters(), lr=lr)

    # optimizer = ChildTuningOptimizer.ChildTuningAdamW(model.parameters(), lr=lr)

    loss_list = []
    acc_list = []
    last_loss = 100
    c = 0
    best_epoch = -1

    last_acc, _, _ = test_dgl(model, X_test, Y_test, ft_test, essay_len_test, device, title=title, is_mask=is_mask)

    for epoch in tqdm(range(epoch_n)):
        total_loss = 0
        gen = utils.batchGenerator_dgl(X_train, Y_train, ft_train, essay_len_train, batch_n, is_random=True)
        i = 0
        model.train()
        for x, y, ft, e_len in gen:

            # print("train")
            # print(type(e_len))
            # print(type(e_len[0]))

            optimizer.zero_grad()  # 将梯度归零

            inputs, labels, tp, e_length = list2tensor_dgl(x, y, ft, e_len, model.p_embd,
                                                           device)  # inputs:(50,34,40,200)

            if is_mask:
                mask = getMask(ft, device)
            else:
                mask = None

            if title:
                result = model(inputs, pos=tp, length_essay=e_length, device=device, mask=mask)[:,
                         1:].contiguous()  # result: (batch_n, doc_l-1, class_n)
                labels = labels[:, 1:].contiguous()
            else:
                result = model(inputs, pos=tp, length_essay=e_length, device=device, mask=mask)

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

        accuracy, dev_aver_loss, _ = test_dgl(model, X_test, Y_test, ft_test, essay_len_test, device, title=title,
                                              is_mask=is_mask)
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

        if (aver_loss > last_loss):
            c += 1
            if c == 10:
                lr = lr * 0.95
                optimizer.param_groups[0]['lr'] = lr
                c = 0
        else:
            c = 0
            last_loss = aver_loss
        torch.save(model, model_dir + '%s_last.pk' % (modelName))

        if (lr < 0.0001) or (aver_loss < 0.5):
            break

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
def test_dgl(model, X, Y, FT, essay_len, device='cpu', batch_n=1, title=False, is_mask=False):
    loss_function = nn.NLLLoss()
    result_list = []
    label_list = []
    model.eval()
    total_loss = 0
    i = 0
    with torch.no_grad():
        # 返回：b_docs, b_labs, b_ft
        # 有一定排序的，先短后长
        gen = utils.batchGenerator_dgl(X, Y, FT, essay_len, batch_n)
        for x, y, ft, e_len in gen:

            # (1,8,40,200)
            # (1,8)
            # (1,8,6)
            # tensor化
            # print("test_dgl")
            # print(type(e_len))
            # print(type(e_len[0]))

            inputs, labels, tp, e_length = list2tensor_dgl(x, y, ft, e_len, model.p_embd, device)

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

                result = model(inputs, pos=tp, length_essay=e_length, device=device, mask=mask)[:,
                         1:].contiguous()  # result: (batch_n, doc_l, class_n)
                # (1,7)
                labels = labels[:, 1:].contiguous()  # labels:(batch_n, doc_l-(title))
            else:
                result = model(inputs, pos=tp, length_essay=e_length, device=device, mask=mask)

            r_n = labels.size()[0] * labels.size()[1]
            # view：把原先tensor中的数据按照行优先的顺序排成一个一维的数据，然后按照参数组合成其他维度的tensor。
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


# def predict(model, x, ft, device='cpu', title=False):
#     inputs, _, tp = list2tensor(x, [], ft, model.p_embd, device)
#
#     if title:
#         result = model(inputs, pos=tp, device=device)[:, 1:].contiguous()
#     else:
#         result = model(inputs, pos=tp, device=device)
#     r_n = result.size()[0] * result.size()[1]
#     result = result.contiguous().view(r_n, -1)
#     return result.cpu().argmax(dim=1).tolist()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Chinese Discourse', usage='newtrain.py [<args>] [-h | --help]')
    parser.add_argument('--model_type', default=2, type=int, help='set model type')
    # 1:POS1, 2:Bottom
    parser.add_argument('--model_name', default='wsj', type=str, help='set model name')
    parser.add_argument('--seed_num', default=1, type=int, help='set seed num')
    parser.add_argument('--epoch', default=700, type=int, help='set epoch num')
    parser.add_argument('--learning_rate', default=0.2, type=float, help='set learning rate')
    parser.add_argument('--dgl_type', default='gcn', type=str, help='set aggregator type of gcn')
    # 'gcn','lstm','pool','mean'
    parser.add_argument('--weight_define', default=1, type=int, help='set how to define weight between nodes')
    # 1:余弦相似度，2:Pearson相似度，3:欧氏距离，4:kendall系数，
    parser.add_argument('--add_self_loop', default=0, type=int, help='whether to add self-loop in dgl')
    # 默认不添加self-loop(是否额外添加自环)
    parser.add_argument('--dgl_layer', default=1, type=int, help='set the number of dgl layers')

    args = parser.parse_args()

    seed_torch(args.seed_num)

    model_package_name = args.model_name
    gcn_aggregator = args.dgl_type
    gcn_weight_id = args.weight_define
    dgl_layers = args.dgl_layer

    in_file = './data/Ch_train.json'

    embed_filename = './embd/new_embeddings2.txt'
    title = True
    max_len = 40

    # 返回：获取本文中每个句子的embedding(单词组合)，每个句子对应的label列表，每个行数据的每个句子的按顺序对应的六个特征，vec_size
    en_documents, en_labels, features, vec_size = utils.getSamplesAndFeatures(in_file, embed_filename, title=title)

    # 返回：按照max_len长度进行处理的句子的embedding(保证每个句子的长度一样了)，每个句子对应的label列表
    pad_documents, pad_labels, essay_length = utils.sentence_padding_dgl(en_documents, en_labels, max_len, vec_size)

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

    tag_model = None

    if args.model_type == 1:
        # 右边的content self attention仍采用原始的sentence_embeeding
        # 左边和右边的采用过了DGL之后的sentence_embeeding
        tag_model = STWithRSbySPP_DGL_POS1(vec_size, hidden_dim, sent_dim, class_n, p_embd=p_embd,
                                           p_embd_dim=p_embd_dim,
                                           pool_type='max_pool', dgl_layer=dgl_layers, gcn_aggr=gcn_aggregator,
                                           weight_id=gcn_weight_id,
                                           loop=args.add_self_loop)
    elif args.model_type == 2:
        # 对原始的sentence_embeeding先进行DGL，剩下的三部分均在此基础上进行
        tag_model = STWithRSbySPP_DGL_POS_Bottom(vec_size, hidden_dim, sent_dim, class_n, p_embd=p_embd,
                                                 p_embd_dim=p_embd_dim,
                                                 pool_type='max_pool', dgl_layer=dgl_layers, gcn_aggr=gcn_aggregator,
                                                 weight_id=gcn_weight_id,
                                                 loop=args.add_self_loop)

    if p_embd == 'embd_b':
        tag_model.posLayer.init_embedding()

    # 创建三个文件名
    if not os.path.exists('./newlog/cn/dgl/' + model_package_name):
        os.mkdir('./newlog/cn/dgl/' + model_package_name)
    if not os.path.exists('./newmodel/cn/dgl/' + model_package_name):
        os.mkdir('./newmodel/cn/dgl/' + model_package_name)
    if not os.path.exists('./newvalue/cn/dgl/' + model_package_name):
        os.mkdir('./newvalue/cn/dgl/' + model_package_name)

    model_dir = './newmodel/cn/dgl/' + model_package_name + '/' + tag_model.getModelName() + '-' + time.strftime(
        '%m-%d_%H.%M', currenttime) + '_seed_' + str(args.seed_num) + '/'
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    print("start Chinese model training")
    starttime = datetime.datetime.now()
    train(tag_model, pad_documents, pad_labels, features, essay_length, is_gpu, epoch_n=args.epoch,
          lr=args.learning_rate, batch_n=batch_n, title=title, is_mask=is_mask)
    endtime = datetime.datetime.now()
    print("本次seed为%d的训练耗时：" % int(args.seed_num))
    print(endtime - starttime)
