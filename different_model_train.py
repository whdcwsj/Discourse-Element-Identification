import sys
import time
import torch
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix
from tensorboardX import SummaryWriter
from torch import optim, nn
import tqdm
from tqdm import *
import os
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix


# 固定随机数种子
def seed_torch(seed=1):
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)   # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)   # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class BertTrainer:
    def __init__(self, config, model, train_data, dev_data, start_time):
        self.config = config
        self.model = model
        self.train_data = train_data
        self.dev_data = dev_data
        self.regular_model_name = 'ch_' + model.getModelName()
        self.currenttime = start_time

        if self.config.add_title:
            self.regular_model_name += '_t'

        self.writer = SummaryWriter(log_dir=self.config.log_path + config.human_model_name + '/cn_' +
                                    self.regular_model_name + '_' + time.strftime('%m-%d_%H.%M', self.currenttime))
        self.model_dir = self.config.model_save_path + self.config.human_model_name + '/' + self.regular_model_name \
                         + '-' + time.strftime('%m-%d_%H.%M', self.currenttime)

        if self.config.cuda:
            self.model.cuda()
            device = 'cuda'
        else:
            self.model.cpu()
            device = 'cpu'

        self.loss_function = nn.NLLLoss()

    def train(self):
        # 冻结模型参数
        # p.requires_grad是返回值
        # 参数p赋值的元素从列表model.parameters()中取。只取param.requires_grad = True(模型参数的可导性是true的元素)
        parameter = filter(lambda p: p.requires_grad, self.model.parameters())

        # optimizer = optim.SGD(parameter, lr=lr)
        # optimizer = optim.Adam(parameter, lr=3e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
        optimizer = optim.Adam(parameter, lr=5e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)

        loss_list = []
        acc_list = []
        last_loss = 100
        c = 0
        best_epoch = -1

        last_acc, _, _ = BertTrainer.test(self)

        for epoch in tqdm(range(self.config.epoch)):
            total_loss = 0
            i = 0
            self.model.train()
            train_data_tqdm = self.train_data
            for data in train_data_tqdm:
                self.model.zero_grad()  # 将梯度归零
                # 获取当前batch的数据信息
                token_ids, pos, labels = data
                labels = labels.squeeze(0)

                if self.config.add_title:
                    result = self.model(documents=token_ids, pos=pos)[:, 1:].contiguous()
                    # result: (batch_n, doc_l-1, class_n)
                    labels = labels[:, 1:].contiguous()
                else:
                    result = self.model(documents=token_ids, pos=pos)

                # 每篇文章下有多个句子，进行规整
                r_n = labels.size()[0] * labels.size()[1]
                result = result.contiguous().view(r_n, -1)
                labels = labels.view(r_n)

                loss = self.loss_function(result, labels)
                loss.backward()  # 反向传播计算得到每个参数的梯度
                optimizer.step()  # 通过梯度下降执行一步参数更新

                total_loss += loss.cpu().detach().numpy()
                i += 1

            aver_loss = total_loss / i
            loss_list.append(aver_loss)

            accuracy, dev_aver_loss, _ = BertTrainer.test(self)
            acc_list.append(accuracy)

            self.writer.add_scalar("loss/train_loss", aver_loss, epoch)
            self.writer.add_scalar("loss/dev_loss", dev_aver_loss, epoch)
            self.writer.add_scalar("performance/accuracy", accuracy, epoch)

            if last_acc < accuracy:
                last_acc = accuracy
                if accuracy > 0.6:
                    # 取每20个epoch中效果最好的
                    torch.save(self.model, self.model_dir + '%s_%d_best.pk' % (self.regular_model_name, int(epoch / 20) * 20))
                if epoch > 200:
                    # 额外记录最好的那一个
                    torch.save(self.model, self.model_dir + '%s_top.pk' % self.regular_model_name)
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

            torch.save(self.model, self.model_dir + '%s_last.pk' % self.regular_model_name)

            # if (lr < 0.0001) or (aver_loss < 0.5):
            #     break

        # 若无最佳模型，跳过该步骤
        if best_epoch == -1:
            pass
        else:
            # top模型文件添加epoch记录
            oldname = self.model_dir + '%s_top.pk' % self.regular_model_name
            newname = self.model_dir + '%s_epoch_%d_top.pk' % (self.regular_model_name, best_epoch)
            os.rename(oldname, newname)

        self.writer.close()

        # plt.cla()
        # plt.plot(range(len(acc_list)), acc_list, range(len(loss_list)), loss_list)
        # plt.legend(['acc_list', 'loss_list'])
        # plt.savefig('../img/' + self.regular_model_name + '.jpg')


    def test(self):

        result_list = []
        label_list = []
        self.model.eval()
        total_loss = 0
        i = 0
        # dev_data_tqdm = tqdm(self.dev_data, desc=r"Test")
        dev_data_tqdm = self.dev_data

        # 冻结参数
        with torch.no_grad():
            for data in dev_data_tqdm:
                # 获取一条数据
                token_ids, pos, labels = data
                labels = labels.squeeze(0)

                if self.config.add_title:
                    result = self.model(documents=token_ids, pos=pos)[:, 1:].contiguous()
                    # result: (batch_n, doc_l, class_n)
                    labels = labels[:, 1:].contiguous()  # labels:(batch_n, doc_l-(title))
                else:
                    result = self.model(documents=token_ids, pos=pos)

                # view：把原先tensor中的数据按照行优先的顺序排成一个一维的数据，然后按照参数组合成其他维度的tensor。
                r_n = labels.size()[0] * labels.size()[1]
                result = result.contiguous().view(r_n, -1)  # result: (doc_l, class_n)  batch_n为1的情况下
                labels = labels.view(r_n)

                loss = self.loss_function(result, labels)
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
