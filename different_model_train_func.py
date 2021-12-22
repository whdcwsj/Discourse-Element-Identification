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


def PRF(a, ignore=[]):
    precision = []
    recall = []
    f1 = []
    real = []
    TP = 0
    TPFP = 0
    TPFN = 0

    for i in range(len(a[0])):
        precision.append(a[i][i] / sum(a[:, i]))
        recall.append(a[i][i] / sum(a[i]))
        f1.append((precision[i] * recall[i] * 2) / (precision[i] + recall[i]))
        real.append(sum(a[i]))
        if i not in ignore:
            TP += a[i][i]
            TPFP += sum(a[:, i])
            TPFN += sum(a[i])

    precision = np.nan_to_num(precision)
    recall = np.nan_to_num(recall)
    f1 = np.nan_to_num(f1)
    real = np.nan_to_num(real)
    print(precision)
    print(recall)
    print(f1)

    a_p = 0
    a_r = 0
    a_f = 0
    m_p = TP / TPFP
    m_r = TP / TPFN

    for i in range(len(f1)):
        if i not in ignore:
            a_p += real[i] * precision[i]
            a_r += real[i] * recall[i]
            a_f += real[i] * f1[i]

    total = sum(real) - sum(real[ignore])
    # print('test', total, a_p)
    print(a_p / total, a_r / total, a_f / total)

    macro_f = np.delete(f1, ignore, 0).mean()
    micro_f = (m_p * m_r * 2) / (m_p + m_r)
    print(macro_f, micro_f)
    # print(m_p, m_r)

    all_prf = [m_r, a_p / total, a_r / total, a_f / total, macro_f, micro_f]
    return precision, recall, f1, all_prf


class BertTrainer:
    def __init__(self, config, model, train_data=None, dev_data=None, test_data=None, start_time=None, add_writer=True,
                 seed=None, list_seed=None):
        self.config = config
        self.model = model
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.regular_model_name = 'ch_' + model.getModelName()
        self.currenttime = start_time
        self.add_writer = add_writer
        if seed is not None:
            self.seed = seed
        self.list_seed = list_seed
        self.summary_file = self.config.value_path + self.config.human_model_name + '/seed_summary.csv'
        self.summary_head = 'seed_num, best_accuracy, best_macro_f1'
        self.csv_head = 'name, accuracy, all-p, all-r, all-f, macro-f, micro-f'
        self.role_name = ['introductionSen', 'thesisSen', 'ideaSen', 'exampleSen', 'conclusionSen', 'otherSen',
                          'evidenceSen']
        self.lr = self.config.lr

        if not os.path.exists(self.config.model_save_path + self.config.human_model_name):
            os.mkdir(self.config.model_save_path + self.config.human_model_name)
        if not os.path.exists(self.config.log_path + self.config.human_model_name):
            os.mkdir(self.config.log_path + self.config.human_model_name)
        if not os.path.exists(self.config.value_path + self.config.human_model_name):
            os.mkdir(self.config.value_path + self.config.human_model_name)

        # 添加每一个小类别的具体结果(precision,recall,F1)
        for n in self.role_name:
            for p in ['-p', '-r', '-f']:
                self.csv_head += ', ' + n + p

        if self.config.add_title:
            self.regular_model_name += '_t'

        if self.add_writer:
            self.writer = SummaryWriter(log_dir=self.config.log_path + config.human_model_name + '/cn_' +
                                    self.regular_model_name + '_' + time.strftime('%m-%d_%H.%M', self.currenttime))

        self.model_store_dir = self.config.model_save_path + self.config.human_model_name + '/'

        if self.currenttime is not None and self.seed is not None:
            self.model_dir = self.config.model_save_path + self.config.human_model_name + '/' + self.regular_model_name\
                             + '-' + time.strftime('%m-%d_%H.%M', self.currenttime) + '_seed_' + str(self.seed) + '/'
            if not os.path.isdir(self.model_dir):
                os.mkdir(self.model_dir)

        if self.config.cuda:
            self.model.cuda()
            device = 'cuda'
        else:
            self.model.cpu()
            device = 'cpu'

        self.loss_function = nn.NLLLoss()

    def train(self):
        if self.seed is not None:
            seed_torch(self.seed)
        # 冻结模型参数
        # p.requires_grad是返回值
        # 参数p赋值的元素从列表model.parameters()中取。只取param.requires_grad = True(模型参数的可导性是true的元素)
        parameter = filter(lambda p: p.requires_grad, self.model.parameters())

        optimizer = optim.SGD(parameter, lr=self.lr)
        # optimizer = optim.Adam(parameter, lr=3e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
        # optimizer = optim.Adam(parameter, lr=5e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)

        loss_list = []
        acc_list = []
        last_loss = 100
        c = 0
        best_epoch = -1

        last_acc, _, _ = BertTrainer.evaluate(self)

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

            accuracy, dev_aver_loss, _ = BertTrainer.evaluate(self)
            acc_list.append(accuracy)

            self.writer.add_scalar("loss/train_loss", aver_loss, epoch)
            self.writer.add_scalar("loss/dev_loss", dev_aver_loss, epoch)
            self.writer.add_scalar("performance/accuracy", accuracy, epoch)

            if last_acc < accuracy:
                last_acc = accuracy
                if accuracy > 0.6:
                    # 取每20个epoch中效果最好的
                    torch.save(self.model, self.model_dir + '%s_%d_best.pk' % (self.regular_model_name, int(epoch / 20) * 20))
                if epoch > 20:
                    # 额外记录最好的那一个
                    torch.save(self.model, self.model_dir + '%s_top.pk' % self.regular_model_name)
                    best_epoch = epoch

            if (aver_loss > last_loss):
                c += 1
                if c == 10:
                    self.lr = self.lr * 0.95
                    optimizer.param_groups[0]['lr'] = self.lr
                    c = 0
            else:
                c = 0
                last_loss = aver_loss

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

        if self.add_writer:
            self.writer.close()

        # plt.cla()
        # plt.plot(range(len(acc_list)), acc_list, range(len(loss_list)), loss_list)
        # plt.legend(['acc_list', 'loss_list'])
        # plt.savefig('../img/' + self.regular_model_name + '.jpg')

    # 验证集代码测试
    def evaluate(self, is_test=False, temp_model=None):
        result_list = []
        label_list = []
        total_loss = 0
        i = 0
        model = self.model
        if is_test:
            data_tqdm = self.test_data
            model = temp_model
        else:
            data_tqdm = self.dev_data

        model.eval()  # 将模型转变为evaluation(测试)模式，排除BN和Dropout对测试的干扰

        # 冻结参数
        with torch.no_grad():
            for data in data_tqdm:
                # 获取一条数据
                token_ids, pos, labels = data
                labels = labels.squeeze(0)

                if self.config.add_title:
                    result = model(documents=token_ids, pos=pos)[:, 1:].contiguous()
                    # result: (batch_n, doc_l, class_n)
                    labels = labels[:, 1:].contiguous()  # labels:(batch_n, doc_l-(title))
                else:
                    result = model(documents=token_ids, pos=pos)

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

        # 横向行是label，纵向列是pred
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

    # 单个seed下所有模型的测试运行
    def ch_test(self, cur_model_dir, cur_seed):

        # 每个CSV文件的名称
        w_file = './newvalue/cn/bert/' + self.config.human_model_name + '/seed_%d.csv' % cur_seed

        with open(w_file, 'w', encoding='utf-8') as wf:
            wf.write(self.csv_head + '\n')
            filenames = os.listdir(cur_model_dir)

            # 保存一下最大的accurancy和macro-f1
            max_accurancy = 0
            max_macro_f1 = 0

            for file in filenames:
                print(file)
                fname = os.path.join(cur_model_dir, file)
                # torch.load()先在CPU上加载，不会依赖于保存模型的设备
                temp_model = torch.load(fname, map_location='cuda')

                accuracy, _, a = BertTrainer.evaluate(self, is_test=True, temp_model=temp_model)

                print(accuracy)
                print(a)

                precision, recall, f1, all_prf = PRF(a[:-1, :-1], ignore=[5])
                accuracy, all_p, all_r, weighted_f, macro_f, micro_f = all_prf

                wf.write('_'.join(file.split('_')[: -1]))
                wf.write(', ' + str(accuracy))
                wf.write(', ' + str(all_p) + ', ' + str(all_r) + ', ' + str(weighted_f))
                wf.write(', ' + str(macro_f))
                wf.write(', ' + str(micro_f))

                if accuracy > max_accurancy:
                    max_accurancy = accuracy
                if macro_f > max_macro_f1:
                    max_macro_f1 = macro_f

                for i in range(len(f1)):
                    wf.write(', ' + str(precision[i]) + ', ' + str(recall[i]) + ', ' + str(f1[i]))

                wf.write('\n')

            # 对应列保存最大的数值
            wf.write(' ')
            wf.write(', ' + str(max_accurancy))
            wf.write(', ' + ', ' + ', ')
            wf.write(', ' + str(max_macro_f1))
            wf.write('\n')

        return max_accurancy, max_macro_f1


    # 测试并汇总所有seed下模型的测试效果
    def test_summary(self):
        i = 0
        # 存储每个seed下的accurancy和macro-f1
        accurancy_list = []
        macro_f1_list = []

        # 避免os.listdir的时候乱序读取
        path_list = os.listdir(self.model_store_dir)
        path_list.sort()
        # print(path_list)

        for seed_model_file in path_list:
            print(self.model_store_dir + seed_model_file)
            temp_dir = self.model_store_dir + seed_model_file
            temp_seed_accu, temp_seed_macro_f1 = BertTrainer.ch_test(self, cur_model_dir=temp_dir, cur_seed=self.list_seed[i])
            accurancy_list.append(temp_seed_accu)
            macro_f1_list.append(temp_seed_macro_f1)
            i = i + 1

        j = 0
        with open(self.summary_file, 'w', encoding='utf-8') as wf:
            wf.write(self.summary_head + '\n')

            for num in self.list_seed:
                wf.write(' seed_' + str(num))
                wf.write(', ' + str(accurancy_list[j]))
                wf.write(', ' + str(macro_f1_list[j]))
                wf.write('\n')
                j = j + 1

            wf.write(' average')
            wf.write(', ' + str(np.mean(accurancy_list)))
            wf.write(', ' + str(np.mean(macro_f1_list)))
            wf.write('\n')




