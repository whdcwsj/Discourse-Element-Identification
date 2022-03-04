import datetime
import os
import logging

from transformers import BertTokenizer
from model import *
from model_gru import *
from model_gate import *
from model_dgl import *
from model_dgl_en import *
import utils_e

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from tqdm import *

from tensorboardX import SummaryWriter
import time
import argparse

import matplotlib.pyplot as plt
plt.switch_backend('Agg')

currenttime = time.localtime()

# model_package_name = 'new_baseline0.67_gate2'
# model_package_name = 'dgl1_p4_lstm_w1_size1'


# 固定随机数种子
def seed_torch(seed=1):
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)   # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)   # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def list2tensor_dgl(x, y, ft, e_len, p_embd, device='cpu'):
    inputs = torch.tensor(x, dtype=torch.long, device=device)
    labels = torch.tensor(y, dtype=torch.long, device=device)

    tp = torch.tensor(ft, dtype=torch.float, device=device)[:, :, :6]
    e_len = torch.tensor(e_len, dtype=torch.long, device=device)
    return inputs, labels, tp, e_len


# 返回：按照max_len长度进行处理的句子的embedding，每个句子对应的label列表，每个行数据的每个句子的按顺序对应的六个特征：['gpos', 'lpos', 'ppos', 'gid', 'lid', 'pid']
# 相比较于中文数据集，多了一个embedding
def train(model, X, Y, FT, essay_len, is_gpu=False, epoch_n=10, lr=0.1, batch_n=100, title=False, embeddings=None, class_n=4):

    modelName = 'e_' + model.getModelName()

    if title:
        modelName += '_t'

    writer = SummaryWriter(
        './newlog/en/dgl/' + model_package_name + '/en_' + modelName + '_' + time.strftime('%m-%d_%H.%M', currenttime)
        + '_seed_' + str(args.seed_num))

    # 20%的数据作为验证集
    X_train, Y_train, ft_train, essay_len_train, X_test, Y_test, ft_test, essay_len_test = utils.dataSplit_dgl(X, Y, FT,
                                                                                                        essay_len, 0.2)

    if(is_gpu):
        model.cuda()
        embeddings.cuda()
        device = 'cuda'
    else:
        model.cpu()
        embeddings.cpu()
        device = 'cpu'

    # 有四类(不算padding的情况下)
    if class_n == 4:
        loss_function = nn.NLLLoss()
    else:
        # 类别范围：[0, C−1]
        # 给nn.NLLLoss的第一个参数weight赋予权重
        # 为每个类指定的手动缩放权重。如果给定，它必须是大小为C的张量。否则，它被视为具有所有张量
        w = torch.tensor([1., 1., 1., 1., 0.], device=device)
        loss_function = nn.NLLLoss(w)

    optimizer = optim.SGD(model.parameters(), lr=lr)

    loss_list = []
    acc_list = []
    last_loss = 100
    c = 0
    best_epoch = -1

    last_acc, _, _ = test_dgl(model, X_test, Y_test, ft_test, essay_len_test, device, title=title,
                              embeddings=embeddings, class_n=class_n)

    for epoch in tqdm(range(epoch_n)):
        total_loss = 0
        gen = utils_e.batchGeneratorId_dgl(X_train, Y_train, ft_train, essay_len_train, batch_n, is_random=True)
        i = 0

        model.train()
        for x, y, ft, e_len in gen:
            # 梯度归零
            optimizer.zero_grad()

            inputs, labels, tp, e_length = list2tensor_dgl(x, y, ft, e_len, model.p_embd, device)   # inputs:(30,18,40)
            # labels:(batch_size, doc_l), tp:(batch_size, doc_l, 6)
            # 将每个embedding的ID号转换为对应的embedding
            inputs = embeddings(inputs)   # (30,18,40,768)

            if title:
                result = model(inputs, pos=tp, length_essay=e_length, device=device)[:, 1:].contiguous()
                labels = labels[:, 1:].contiguous()
            else:
                result = model(inputs, pos=tp, length_essay=e_length, device=device)

            r_n = labels.size()[0]*labels.size()[1]
            result = result.contiguous().view(r_n, -1)
            labels = labels.view(r_n)
            loss = loss_function(result, labels)

            loss.backward()
            optimizer.step()
            total_loss += loss.cpu().detach().numpy()
            i += 1

        aver_loss = total_loss/i
        loss_list.append(aver_loss)

        accuracy, dev_aver_loss, _ = test_dgl(model, X_test, Y_test, ft_test, essay_len_test, device, title=title,
                                              embeddings=embeddings, class_n=class_n)
        acc_list.append(accuracy)

        writer.add_scalar("en_loss/train", aver_loss, epoch)
        writer.add_scalar("en_loss/dev", dev_aver_loss, epoch)
        writer.add_scalar("en_performance/accuracy", accuracy, epoch)

        if last_acc < accuracy:
            last_acc = accuracy
            if accuracy > 0.65:
                torch.save(model, model_dir + '%s_%d_best.pk' % (modelName, int(epoch/20)*20))
            if epoch > 100:
                # 额外记录最好的那一个
                torch.save(model, model_dir + '%s_top.pk' % modelName)
                best_epoch = epoch


        if(aver_loss > last_loss):
            c += 1
            if c == 10:
                lr = lr * 0.95
                optimizer.param_groups[0]['lr'] = lr
                c = 0
        else:
            c = 0
            last_loss = aver_loss

        torch.save(model, model_dir + '%s_last.pk' % (modelName))

        if(lr < 0.0001):
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
    #
    # plt.savefig('./img/'+modelName+'.jpg')

# 有embeddings，title=True
def test_dgl(model, X, Y, FT, essay_len, device='cpu', batch_n=1, title=False, embeddings=None, class_n=4):

    loss_function = nn.NLLLoss()
    result_list = []
    label_list = []
    # model.eval()的作用是不启用Batch Normalization和Dropout
    model.eval()
    total_loss = 0
    i = 0

    with torch.no_grad():
        gen = utils_e.batchGeneratorId_dgl(X, Y, FT, essay_len, batch_n)
        for x, y, ft, e_len in gen:

            # (1,10,40)
            # (1,10)   tensor([[0, 4, 4, 1, 2, 3, 2, 3, 1, 2]], device='cuda:0')
            # (1,10,6)
            inputs, labels, tp, e_length = list2tensor_dgl(x, y, ft, e_len, model.p_embd, device)
            inputs = embeddings(inputs)   # (batch_size, dol_l, sen_l, 768)

            if title:
                # (1,9,5)
                # 使用transpose或permute进行维度变换后，调用contiguous，然后方可使用view对维度进行变形
                result = model(inputs, pos=tp, length_essay=e_length, device=device)[:, 1:].contiguous()   # result: (batch_n, doc_l-1, class_n)
                # (1,9)
                labels = labels[:, 1:].contiguous()
            else:
                result = model(inputs, pos=tp, length_essay=e_length, device=device)

            r_n = labels.size()[0]*labels.size()[1]
            # (9，5)
            result = result.contiguous().view(r_n, -1)
            # (9,)
            labels = labels.view(r_n)

            loss = loss_function(result, labels)
            total_loss += loss.cpu().detach().numpy()
            i += 1

            result_list.append(result)
            label_list.append(labels)

    aver_loss = total_loss / i

    # (1059,5)
    preds = torch.cat(result_list)
    # (1059,)
    labels = torch.cat(label_list)
    t_c = 0
    d = preds.size(-1)  # d = 5
    # 混淆矩阵
    if class_n == 4:
        a = np.zeros((d, d))
    else:
        a = np.zeros((d-1, d-1))
    l = preds.size(0)  # l = 1059
    for i in range(l):
        if class_n == 4:
            p = preds[i][:].cpu().argmax().numpy()
            r = int(labels[i].cpu().numpy())
            a[r][p] += 1
            if p == r:
                t_c += 1
        else:
            # 跳过最后一个标签
            p = preds[i][:-1].cpu().argmax().numpy()
            r = int(labels[i].cpu().numpy())
            if r != 4:
                a[r][p] += 1
                if p == r:
                    t_c += 1
    accuracy = t_c / l

    # class_n = 4 的情况下
    # 0号是padding
    # print(a)
    # [[  0.   0.   0.   0.   0.]
    #  [  0.   0.   0.  85.  27.]
    #  [  0.   0.   0. 208.  14.]
    #  [  0.   0.   0. 557.   5.]
    #  [  0.   0.   0.  59. 104.]]

    return accuracy, aver_loss, a




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='English Discourse', usage='newtrain_en.py [<args>] [-h | --help]')
    parser.add_argument('--model_name', default='wsj', type=str, help='set model name')
    parser.add_argument('--seed_num', default=1, type=int, help='set seed num.')
    parser.add_argument('--epoch', default=1500, type=int, help='set epoch num')
    parser.add_argument('--learning_rate', default=0.1, type=float, help='set learning rate')
    parser.add_argument('--dgl_type', default='lstm', type=str, help='set aggregator type of SAGEConv')
    # 'gcn','lstm','pool','mean'
    parser.add_argument('--weight_define', default=1, type=int, help='set how to define weight between nodes'
                                                                     ' in SAGEConv')
    # 1:余弦相似度，2:Pearson相似度，3:欧氏距离，4:kendall系数，
    parser.add_argument('--add_self_loop', default=0, type=int, help='whether to add self-loop in dgl')
    # 默认不添加self-loop(是否额外添加自环)
    parser.add_argument('--dgl_layer', default=1, type=int, help='set the number of dgl layers')
    parser.add_argument('--window_size', default=1, type=int, help='set the size of dgl sliding window')

    args = parser.parse_args()

    seed_torch(args.seed_num)

    model_package_name = args.model_name
    gcn_aggregator = args.dgl_type
    gcn_weight_id = args.weight_define
    dgl_layers = args.dgl_layer

    in_file = './data/En_train.json'
    is_word=False

    print('load Bert Tokenizer...')
    BERT_PATH = './bert/bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(BERT_PATH)

    title = True
    max_len = 40

    # 获取本文中每个句子的embedding的id号；每个句子对应的label列表；每个行数据的每个句子的按顺序对应的六个特征：['gpos', 'lpos', 'ppos', 'gid', 'lid', 'pid']
    en_documents, en_labels, features = utils_e.getEnglishSamplesBertId(in_file, tokenizer, title=title, is_word=is_word)
    # print("原本的：")
    # print(en_documents[-1])
    # print(en_labels[-1])

    # 返回：按照max_len长度进行处理的句子的embedding的id号，每个句子对应的label列表
    pad_documents, pad_labels, essay_length = utils_e.sentencePaddingId_dgl(en_documents, en_labels, max_len)
    # print("Pad之后的：")
    # print(pad_documents[-1])
    # print(pad_labels[-1])

    # [[2336, 2323, 5702, 2524, 2030, 2652, 4368, 1029, 2119, 2064, 5335, 2037, 2925],
    #  [2070, 2111, 2903, 2008, 5702, 2524, 2003, 6827, 2112, 2005, 2336, 1010, 4728, 2500, 2111, 2228, 2008, 2652, 4368,
    #   2003, 5949, 1997, 2051, 1012],
    #
    # [0, 4, 1, 2, 3, 4, 3, 3, 3, 2, 4, 3, 3, 3, 3, 3, 3, 3, 1]
    #
    # [[2336, 2323, 5702, 2524, 2030, 2652, 4368, 1029, 2119, 2064, 5335, 2037, 2925, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #  [2070, 2111, 2903, 2008, 5702, 2524, 2003, 6827, 2112, 2005, 2336, 1010, 4728, 2500, 2111, 2228, 2008, 2652, 4368,
    #   2003, 5949, 1997, 2051, 1012, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #
    # [0, 4, 1, 2, 3, 4, 3, 3, 3, 2, 4, 3, 3, 3, 3, 3, 3, 3, 1]

    batch_n = 30

    is_gpu = True
    if is_gpu and torch.cuda.is_available():
        pass
    else:
        is_gpu = False

    hidden_dim = 64
    sent_dim = 64

    p_embd = 'add'
    pos_dim = 0
    p_embd_dim=16
    if p_embd in ['embd_b', 'embd_c']:
        p_embd_dim = hidden_dim*2

    # bert_model的embedding的讲解：https://zhuanlan.zhihu.com/p/360988428
    embeddings = torch.load('./embd/bert-base-uncased-word_embeddings.pkl')

    # embeddings.embedding_dim：768
    # Main Claim(主要声明)，Claim(声明)，Premise(前提)，Other，Padding
    # 声明模型的MLP的class_n=5

    tag_model = EnSTWithRSbySPP_DGL_Bottom_Sliding_Window(embeddings.embedding_dim, hidden_dim, sent_dim, class_n=5,
                                                          p_embd=p_embd,
                                                          p_embd_dim=p_embd_dim,
                                                          pool_type='max_pool',
                                                          dgl_layer=dgl_layers,
                                                          gcn_aggr=gcn_aggregator,
                                                          weight_id=gcn_weight_id,
                                                          loop=args.add_self_loop,
                                                          window_size=args.window_size)

    # tag_model = EnSTWithRSbySPP_GRU(embeddings.embedding_dim, hidden_dim, sent_dim, class_n=5, p_embd=p_embd,
    #                           p_embd_dim=p_embd_dim, pool_type='max_pool')

    # tag_model = EnSTWithRSbySPP_GATE(embeddings.embedding_dim, hidden_dim, sent_dim, class_n=5, p_embd=p_embd,
    #                           p_embd_dim=p_embd_dim, pool_type='max_pool')

    # 创建三个文件名
    if not os.path.exists('./newlog/en/dgl/' + model_package_name):
        os.mkdir('./newlog/en/dgl/' + model_package_name)
    if not os.path.exists('./newmodel/en/dgl/' + model_package_name):
        os.mkdir('./newmodel/en/dgl/' + model_package_name)
    if not os.path.exists('./newvalue/en/dgl/' + model_package_name):
        os.mkdir('./newvalue/en/dgl/' + model_package_name)

    # 这个是指不考虑padding的情况下
    class_n = 4
    model_dir = './newmodel/en/dgl/' + model_package_name + '/' + tag_model.getModelName() + '-' + time.strftime(
        '%m-%d_%H.%M', currenttime) + '_seed_' + str(args.seed_num) + '/'
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    print("start English model training")
    starttime = datetime.datetime.now()

    # 训练的时候class_n设置为4
    # 在与Stab和Gurevych（2017）进行比较时， class_n为3：Main Claim(主要声明)，Claim(声明)，Premise(前提)
    train(tag_model, pad_documents, pad_labels, features, essay_length, is_gpu, epoch_n=args.epoch,
          lr=args.learning_rate, batch_n=batch_n, title=title, embeddings=embeddings, class_n=class_n)
    endtime = datetime.datetime.now()
    print("本次seed为%d的训练耗时：" % int(args.seed_num))
    print(endtime - starttime)

