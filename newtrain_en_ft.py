import datetime
import os

from transformers import BertTokenizer
# import config
from model import *
from model_gru import *
from model_dgl_enft import *
import utils_e

import numpy as np

import argparse
from tensorboardX import SummaryWriter
import time

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from tqdm import *

import matplotlib.pyplot as plt

plt.switch_backend('Agg')

currenttime = time.localtime()

# model_package_name = 'wsj'


def list2tensor_dgl(x, y, ft, e_len, p_embd, device='cpu'):
    inputs = torch.tensor(x, dtype=torch.long, device=device)
    labels = torch.tensor(y, dtype=torch.long, device=device)
    tp = torch.tensor(ft, dtype=torch.float, device=device)[:, :, :6]
    tft = torch.tensor(ft, dtype=torch.float, device=device)[:, :, 6:]
    e_len = torch.tensor(e_len, dtype=torch.long, device=device)
    return inputs, labels, tp, tft, e_len

# 固定随机数种子
def seed_torch(seed=1):
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)   # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)   # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train(model, X, Y, FT, essay_len, is_gpu=False, epoch_n=10, lr=0.1, batch_n=100, title=False, embeddings=None):

    modelName = 'e_' + model.getModelName()

    if title:
        modelName += '_t'

    writer = SummaryWriter(
        './newlog/enft/dgl/' + model_package_name + '/en_ft_' + modelName + '_' + time.strftime(
        '%m-%d_%H.%M', currenttime) + '_seed_' + str(args.seed_num))

    # 20%的数据作为验证集
    X_train, Y_train, ft_train, essay_len_train, X_test, Y_test, ft_test, essay_len_test = utils.dataSplit_dgl(X, Y, FT,
                                                                                                        essay_len, 0.2)

    if (is_gpu):
        model.cuda()
        embeddings.cuda()
        device = 'cuda'
    else:
        model.cpu()
        embeddings.cpu()
        device = 'cpu'

    # with feature的class_n是3种(不考虑Others)
    # 在可变长度序列中，填充序列并使用ignore_index引作为填充目标索引，以避免考虑填充值（来自输入和目标）。
    # 如果有n个类，必须为logit维度准备（n+1）个类（交叉熵损失的输入），包括pad类，然后使用ignore_index选项忽略它。
    # 指定一个目标值，该目标值将被忽略并且不会对输入梯度产生影响
    loss_function = nn.NLLLoss(ignore_index=4)

    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_list = []
    acc_list = []
    last_loss = 100
    c = 0
    best_epoch = -1

    last_acc, _, _ = test_dgl(model, X_test, Y_test, ft_test, essay_len_test, device, title=title,
                              embeddings=embeddings)
    # acc_list.append(last_acc)
    # last_acc = max(0.6, last_acc)

    for epoch in tqdm(range(epoch_n)):
        total_loss = 0
        gen = utils_e.batchGeneratorId_dgl(X_train, Y_train, ft_train, essay_length=essay_len_train, batch_n=batch_n, is_random=True)
        i = 0

        model.train()
        for x, y, ft, e_len in gen:

            optimizer.zero_grad()

            inputs, labels, tp, tft, e_length = list2tensor_dgl(x, y, ft, e_len, model.p_embd, device)
            inputs = embeddings(inputs)

            if title:
                result = model(inputs, tp, tft, length_essay=e_length, device=device)[:, 1:].contiguous()
                labels = labels[:, 1:].contiguous()
            else:
                result = model(inputs, tp, tft, length_essay=e_length, device=device)

            r_n = labels.size()[0] * labels.size()[1]
            result = result.contiguous().view(r_n, -1)
            labels = labels.view(r_n)
            loss = loss_function(result, labels)

            loss.backward()
            optimizer.step()
            total_loss += loss.cpu().detach().numpy()
            i += 1

        aver_loss = total_loss / i
        loss_list.append(aver_loss)

        # if epoch % 10 == 0:
        accuracy, dev_aver_loss, _ = test_dgl(model, X_test, Y_test, ft_test, essay_len_test, device, title=title,
                                              embeddings=embeddings)
        acc_list.append(accuracy)

        writer.add_scalar("en_feature_loss/train", aver_loss, epoch)
        writer.add_scalar("en_feature_loss/dev", dev_aver_loss, epoch)
        writer.add_scalar("en_feature_performance/accuracy", accuracy, epoch)

        if last_acc < accuracy:
            last_acc = accuracy
            if accuracy > 0.77:
                torch.save(model, model_dir + '%s_%d_best.pk' % (modelName, int(epoch / 10) * 10))
            if epoch > 100:
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

        if (lr < 0.0001):
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
    # plt.savefig('./img/' + modelName + '.jpg')

    # plt.show()


def test_dgl(model, X, Y, FT, essay_len, device='cpu', batch_n=1, title=False, embeddings=None):

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

            inputs, labels, tp, tft, e_length = list2tensor_dgl(x, y, ft, e_len, model.p_embd, device)
            inputs = embeddings(inputs)

            if title:
                result = model(inputs, tp, tft, length_essay=e_length, device=device)[:, 1:].contiguous()
                labels = labels[:, 1:].contiguous()
            else:
                result = model(inputs, tp, tft, length_essay=e_length, device=device)

            r_n = labels.size()[0] * labels.size()[1]
            result = result.contiguous().view(r_n, -1)
            labels = labels.view(r_n)

            loss = loss_function(result, labels)
            total_loss += loss.cpu().detach().numpy()
            i += 1

            result_list.append(result)
            label_list.append(labels)

    aver_loss = total_loss / i

    preds = torch.cat(result_list)
    labels = torch.cat(label_list)
    t_c = 0
    d = preds.size(-1)
    a = np.zeros((d - 1, d - 1))
    l = 0
    for i in range(preds.size(0)):
        p = preds[i][:-1].cpu().argmax().numpy()
        r = int(labels[i].cpu().numpy())
        if r != 4:
            # if True:
            a[r][p] += 1
            l += 1
            if p == r:
                t_c += 1
    accuracy = t_c / l

    return accuracy, aver_loss, a


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='English Discourse', usage='newtrain_en_ft.py [<args>] [-h | --help]')
    parser.add_argument('--model_name', default='wsj', type=str, help='set model name')
    parser.add_argument('--seed_num', default=1, type=int, help='Set seed num.')
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
    is_word = False
    class_n = 5

    print('load Bert Tokenizer...')
    BERT_PATH = './bert/bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(BERT_PATH)

    title = True
    max_len = 40

    en_documents, en_labels, features = utils_e.getEnglishSamplesBertId(in_file, tokenizer, title=title, is_word=is_word)

    pad_documents, pad_labels, essay_length = utils_e.sentencePaddingId_dgl(en_documents, en_labels, max_len)


    # for i in range(len(essay_length)):
    #     if isinstance(essay_length[i], int):
    #         pass
    #     else:
    #         print("wrong")
    # print("correct")


    # 处理新增的手工特征
    n_features = utils_e.featuresExtend(features, en_documents, en_labels, tokenizer)
    ft_size = len(n_features[0][0]) - 7

    batch_num = 30

    is_gpu = True
    if is_gpu and torch.cuda.is_available():
        pass
    else:
        is_gpu = False

    hidden_dim = 64
    sent_dim = 64

    p_embd = 'add'
    pos_dim = 0
    p_embd_dim = 16
    if p_embd in ['embd_b', 'embd_c']:
        p_embd_dim = hidden_dim * 2

    embeddings = torch.load('./embd/bert-base-uncased-word_embeddings.pkl')
    # embeddings.weight.requires_grad = True

    # tag_model = torch.load('./model/STE_model_128_128_last.pk')
    # 分类器依然是五分类器
    tag_model = EnSTWithRSbySPPWithFt2_GRU_DGL_Bottom_Sliding_Window(embeddings.embedding_dim, hidden_dim, sent_dim,
                                                                     class_n,
                                                                     p_embd=p_embd,
                                                                     p_embd_dim=p_embd_dim,
                                                                     ft_size=ft_size,
                                                                     pool_type='max_pool',
                                                                     dgl_layer=dgl_layers,
                                                                     gcn_aggr=gcn_aggregator,
                                                                     weight_id=gcn_weight_id,
                                                                     loop=args.add_self_loop,
                                                                     window_size=args.window_size)

    # 创建三个文件名
    if not os.path.exists('./newlog/enft/dgl/' + model_package_name):
        os.mkdir('./newlog/enft/dgl/' + model_package_name)
    if not os.path.exists('./newmodel/enft/dgl/' + model_package_name):
        os.mkdir('./newmodel/enft/dgl/' + model_package_name)
    if not os.path.exists('./newvalue/enft/dgl/' + model_package_name):
        os.mkdir('./newvalue/enft/dgl/' + model_package_name)


    model_dir = './newmodel/enft/dgl/' + model_package_name + '/' + tag_model.getModelName() + '-' + time.strftime(
        '%m-%d_%H.%M', currenttime) + '_seed_' + str(args.seed_num) + '/'
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    print("start English with feature model training")
    starttime = datetime.datetime.now()
    train(tag_model, pad_documents, pad_labels, n_features, essay_length, is_gpu, epoch_n=args.epoch,
          lr=args.learning_rate, batch_n=batch_num, title=title, embeddings=embeddings)
    endtime = datetime.datetime.now()
    print("本次seed为%d的训练耗时：" % int(args.seed_num))
    print(endtime - starttime)

