import datetime
import os

from transformers import BertTokenizer
# import config
from model import STWithRSbySPPWithFt2
import utils_e as utils

import numpy as np

import argparse
from tensorboardX import SummaryWriter
import time

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

plt.switch_backend('Agg')

currenttime = time.localtime()

model_package_name = 'baseline0.77_drop0.1'


def list2tensor(x, y, ft, p_embd, device='cpu'):
    # print([len(j) for i in x[:3] for j in i])
    inputs = torch.tensor(x, dtype=torch.long, device=device)
    labels = torch.tensor(y, dtype=torch.long, device=device)

    tp = torch.tensor(ft, dtype=torch.float, device=device)[:, :, :6]

    tft = torch.tensor(ft, dtype=torch.float, device=device)[:, :, 6:]
    return inputs, labels, tp, tft

# 固定随机数种子
def seed_torch(seed=1):
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)   # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)   # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train(model, X, Y, FT, is_gpu=False, epoch_n=10, lr=0.1, batch_n=100, title=False, embeddings=None):

    modelName = 'e_' + model.getModelName()

    if title:
        modelName += '_t'

    writer = SummaryWriter('./newlog/enft/' + model_package_name + '/en_ft_' + modelName + '_' + time.strftime('%m-%d_%H.%M', currenttime))

    X_train, Y_train, ft_train, X_test, Y_test, ft_test = utils.dataSplit(X, Y, FT, 0.2)

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

    last_acc, _ = test(model, X_test, Y_test, ft_test, device, title=title, embeddings=embeddings)
    # acc_list.append(last_acc)
    # last_acc = max(0.6, last_acc)

    for epoch in range(epoch_n):
        total_loss = 0
        gen = utils.batchGeneratorId(X_train, Y_train, ft_train, batch_n, is_random=True)
        i = 0

        for x, y, ft in gen:
            optimizer.zero_grad()

            inputs, labels, tp, tft = list2tensor(x, y, ft, model.p_embd, device)
            inputs = embeddings(inputs)

            if title:
                result = model(inputs, tp, tft, device=device)[:, 1:].contiguous()
                labels = labels[:, 1:].contiguous()
            else:
                result = model(inputs, tp, tft, device=device)

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
        accuracy, _ = test(model, X_test, Y_test, ft_test, device, title=title, embeddings=embeddings)
        acc_list.append(accuracy)

        writer.add_scalar("en_feature_loss/train", aver_loss, epoch)
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


def test(model, X, Y, FT, device='cpu', batch_n=1, title=False, embeddings=None):
    result_list = []
    label_list = []
    with torch.no_grad():
        gen = utils.batchGeneratorId(X, Y, FT, batch_n)
        for x, y, ft in gen:

            inputs, labels, tp, tft = list2tensor(x, y, ft, model.p_embd, device)
            inputs = embeddings(inputs)

            if title:
                result = model(inputs, tp, tft, device=device)[:, 1:].contiguous()
                labels = labels[:, 1:].contiguous()
            else:
                result = model(inputs, tp, tft, device=device)

            r_n = labels.size()[0] * labels.size()[1]
            result = result.contiguous().view(r_n, -1)
            labels = labels.view(r_n)

            result_list.append(result)
            label_list.append(labels)

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

    return accuracy, a


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='English Discourse', usage='newtrain_en_ft.py [<args>] [-h | --help]')
    parser.add_argument('--seed_num', default=1, type=int, help='Set seed num.')
    args = parser.parse_args()

    seed_torch(args.seed_num)

    in_file = './data/En_train.json'
    is_word = False
    class_n = 5

    print('load Bert Tokenizer...')
    BERT_PATH = '/home/wsj/bert_model/bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(BERT_PATH)

    title = True
    max_len = 40

    en_documents, en_labels, features = utils.getEnglishSamplesBertId(in_file, tokenizer, title=title, is_word=is_word)

    pad_documents, pad_labels = utils.sentencePaddingId(en_documents, en_labels, max_len)

    n_features = utils.featuresExtend(features, en_documents, en_labels, tokenizer)
    ft_size = len(n_features[0][0]) - 7

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
    p_embd_dim = 16
    if p_embd in ['embd_b', 'embd_c']:
        p_embd_dim = hidden_dim * 2

    embeddings = torch.load('./embd/bert-base-uncased-word_embeddings.pkl')
    # embeddings.weight.requires_grad = True

    # tag_model = torch.load('./model/STE_model_128_128_last.pk')
    tag_model = STWithRSbySPPWithFt2(embeddings.embedding_dim, hidden_dim, sent_dim, class_n, p_embd=p_embd,
                                     p_embd_dim=p_embd_dim, ft_size=ft_size)

    # 创建三个文件名
    if not os.path.exists('./newlog/enft/' + model_package_name):
        os.mkdir('./newlog/enft/' + model_package_name)
    if not os.path.exists('./newmodel/enft/' + model_package_name):
        os.mkdir('./newmodel/enft/' + model_package_name)
    if not os.path.exists('./newvalue/enft/' + model_package_name):
        os.mkdir('./newvalue/enft/' + model_package_name)


    model_dir = './newmodel/enft/' + model_package_name + '/' + tag_model.getModelName() + '-' + time.strftime('%m-%d_%H.%M', currenttime) + '_seed_' + str(args.seed_num) + '/'
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)


    print("start English with feature model training")
    starttime = datetime.datetime.now()
    train(tag_model, pad_documents, pad_labels, n_features, is_gpu, epoch_n=1500, lr=0.1, batch_n=batch_n, title=title,
          embeddings=embeddings)
    endtime = datetime.datetime.now()
    print("本次训练耗时：")
    print(endtime - starttime)

