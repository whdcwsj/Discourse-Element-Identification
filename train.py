import utils
from model import *

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import datetime
import os
import logging

import matplotlib.pyplot as plt
plt.switch_backend('Agg')

datastr = datetime.datetime.now().strftime('%y%m%d%H%M%S')

# logging模块函数，format指定输出的格式和内容，datafmt指定时间格式（同time.strftime）
# filename指定日志文件名，filemode指定日志文件的打开模式，w或a

# logging函数根据它们用来跟踪的事件的级别或严重程度来命名
# 默认Level为warning，这个等级及以上的信息才会输出
# CRITICAL:50；ERROR:40；WARNING:30；INFO=20；DEBUG=10；NOTSET=0

logging.basicConfig(level=logging.INFO,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename='./log/cn_sent_tag_%s.log' % datastr,
                filemode='w')

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
    # 10%的数据作为验证集
    X_train, Y_train, ft_train, X_test, Y_test, ft_test = utils.dataSplit(X, Y, FT, 0.1)
    
    if(is_gpu):
        model.cuda()
        device = 'cuda'
    else:
        model.cpu()
        device = 'cpu'
        
    modelName = model.getModelName()
    if title:
        modelName += '_t'  
    logging.info(modelName)

    # nn.NLLLoss输入是一个对数概率向量和一个目标标签
    # 用于训练一个N类分类器

    # 先softmax，范围(0,1)
    # input=torch.randn(3,3)
    # soft_input=torch.nn.Softmax(dim=0) 按列
    # soft_input(input)
    # tensor([[0.7284, 0.7364, 0.3343],
    #         [0.1565, 0.0365, 0.0408],
    #         [0.1150, 0.2270, 0.6250]])
    # 在Log_softmax，在softmax之后进行一次Log，范围(-∞，+∞)
    # torch.log(soft_input(input))
    # tensor([[-0.3168, -0.3059, -1.0958],
    #         [-1.8546, -3.3093, -3.1995],
    #         [-2.1625, -1.4827, -0.4701]])
    # nn.NLLLoss的结果就是把上面的输出与Label对应的那个值拿出来，再去掉负号，再求均值
    # 假设标签是[0,1,2]，第一行取第0个元素，第二行取第1个，第三行取第2个，去掉负号，即[0.3168,3.3093,0.4701],求平均值
    # target=torch.tensor([0,1,2])
    # loss(input,target)
    # tensor(0.1365)

    loss_function = nn.NLLLoss()

    # 可以尝试一下不同的优化器
    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_list = []
    acc_list = []
    last_loss = 100
    c = 0
    c1 = 0

    last_acc, _ = test(model, X_test, Y_test, ft_test, device, title=title, is_mask=is_mask)
    logging.info('first acc: %f' % last_acc)  
    for epoch in range(epoch_n):
        total_loss = 0
        gen = utils.batchGenerator(X_train, Y_train, ft_train, batch_n, is_random=True)
        i = 0
        model.train()
        for x, y, ft in gen:
            optimizer.zero_grad()  # 将梯度归零
            
            inputs, labels, tp = list2tensor(x, y, ft, model.p_embd, device)
            
            if is_mask:
                mask = getMask(ft, device)
            else:
                mask = None

            if title:
                result = model(inputs, pos=tp, device=device, mask=mask)[:, 1:].contiguous()
                labels = labels[:, 1:].contiguous()
            else:
                result = model(inputs, pos=tp, device=device, mask=mask)
            

            r_n = labels.size()[0]*labels.size()[1]
            result = result.contiguous().view(r_n, -1)
            labels = labels.view(r_n)
            
            loss = loss_function(result, labels)
            loss.backward()   # 反向传播计算得到每个参数的梯度
            optimizer.step()   # 通过梯度下降执行一步参数更新
            
            total_loss += loss.cpu().detach().numpy()
            i += 1
            
        aver_loss = total_loss/i
        loss_list.append(aver_loss)

        accuracy, _ = test(model, X_test, Y_test, ft_test, device, title=title, is_mask=is_mask)
        acc_list.append(accuracy)
        if last_acc < accuracy:
            last_acc = accuracy
            if accuracy > 0.58:
                torch.save(model, model_dir + '%s_%d_best.pk' % (modelName, int(epoch/20)*20))
        logging.info('%d total loss: %f accuracy: %f' % (epoch, aver_loss, accuracy))

        # print("------------------")
        # for i in optimizer.param_groups:
        #     print(i.keys())
        # break

        # pytorch的动态学习率
        # loss数值每超过100大于10次，lr就下降5%
        # optimizer.param_groups[0]是个dict
        # dict_keys(['params', 'lr', 'momentum', 'dampening', 'weight_decay', 'nesterov'])
        if(aver_loss > last_loss):
            c += 1
            if c == 10:
                lr = lr * 0.95
                optimizer.param_groups[0]['lr'] = lr
                logging.info('lr: %f' % lr)
                c = 0
        else:
            c = 0
            last_loss = aver_loss
        torch.save(model, model_dir + '%s_last.pk' % (modelName))

        if(lr < 0.0001) or (aver_loss < 0.5):
            break
    plt.cla()
    plt.plot(range(len(acc_list)), acc_list, range(len(loss_list)), loss_list)
    plt.legend(['acc_list', 'loss_list'])
    plt.savefig('./img/'+modelName+'.jpg')

# 训练集数据的10%，作为验证集
# X=按照max_len长度进行处理的句子的embedding，Y=每个句子对应的label列表，FT=每个行数据的每个句子的按顺序对应的六个特征
def test(model, X, Y, FT, device='cpu', batch_n=1, title=False, is_mask=False):
    result_list = []
    label_list = []
    model.eval()
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

            r_n = labels.size()[0]*labels.size()[1]
            # view：把原先tensor中的数据按照行优先的顺序排成一个一维的数据，然后按照参数组合成其他维度的tensor。
            result = result.contiguous().view(r_n, -1)  # result: (doc_l, class_n)  batch_n为1的情况下
            # label变成一维的
            labels = labels.view(r_n)

            result_list.append(result)
            label_list.append(labels)

    preds = torch.cat(result_list)   # preds:(2866,8)
    labels = torch.cat(label_list)   # labels:(2866,)
    t_c = 0
    # 混淆矩阵
    a = np.zeros((8, 8))
    l = preds.size()[0]
    for i in range(l):
        p = preds[i].cpu().argmax().numpy()
        r = int(labels[i].cpu().numpy())
        a[r][p] += 1
        if p == r:
            t_c += 1
    accuracy = t_c / l
    return accuracy, a   
    
def predict(model, x, ft, device='cpu', title=False):
    inputs, _, tp = list2tensor(x, [], ft, model.p_embd, device)
                
    if title:
        result = model(inputs, pos=tp, device=device)[:, 1:].contiguous()
    else:
        result = model(inputs, pos=tp, device=device)
    r_n = result.size()[0]*result.size()[1]
    result = result.contiguous().view(r_n, -1)
    return result.cpu().argmax(dim=1).tolist()

    
if __name__ == "__main__":

    in_file = './data/Ch_train.json'

    logging.info(in_file)
    
    embed_filename = './embd/new_embeddings2.txt'
    title = True
    max_len = 40
    is_topic = False
    is_suppt = False

    # 返回：获取本文中每个句子的embedding(单词组合)，每个句子对应的label列表，每个行数据的每个句子的按顺序对应的六个特征，vec_size
    en_documents, en_labels, features, vec_size = utils.getSamplesAndFeatures(in_file, embed_filename, title=title)

    # 返回：按照max_len长度进行处理的句子的embedding，每个句子对应的label列表
    pad_documents, pad_labels = utils.sentence_padding(en_documents, en_labels, max_len, vec_size)
    
    is_mask = False

    # Introduction，Thesis，Main Idea，Evidence，Elaboration，Conclusion，Other
    # ？？？哪来的八种情况
    # 还有Padding
    class_n = 8
    batch_n = 50
    is_gpu = True

    if is_gpu and torch.cuda.is_available():
        is_gpu = True
    else:
        is_gpu = False
    
    hidden_dim = 128
    sent_dim = 128
    
    p_embd = 'add'
    p_embd_dim=16
    
    if p_embd in ['embd_b', 'embd_c', 'addv']:
        p_embd_dim = hidden_dim*2
            
    if p_embd != 'embd_c':
        # 将后面三个'gid', 'lid', 'pid'根据前面三个'gpos'*40, 'lpos'*20, 'ppos'*10分别扩大
        # 返回大于等于该数值的最小整数
        features = utils.discretePos(features)

    # sent_dim用于句间ateention的维度dk的吗
    tag_model = STWithRSbySPP(vec_size, hidden_dim, sent_dim, class_n, p_embd=p_embd, p_embd_dim=p_embd_dim, pool_type='max_pool')
    
    
    if p_embd == 'embd_b':
        tag_model.posLayer.init_embedding()
    
    model_dir = './model/roles/%s_%s/' % (tag_model.getModelName(), datastr)
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    
    logging.info('start training model...')
    starttime = datetime.datetime.now()
    train(tag_model, pad_documents, pad_labels, features, is_gpu, epoch_n=700, lr=0.2, batch_n=batch_n, title=title, is_mask=is_mask)
    endtime = datetime.datetime.now()
    logging.info(endtime - starttime)

